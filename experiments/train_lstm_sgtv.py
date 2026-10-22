"""Train the LSTM + SGTV reference condition for the CPP memory study.

This is deliberately narrower than the upstream paper runner: no DTM, no A*
intervention, no offline/full-map environment, and no alternate reward terms.
The TTT runner must use the same environment construction and PPO settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.common import HybridLocalGlobalEncoderConfig, RobotStateEncoderConfig
from learning.observation import HybridLocalGlobalCPPObservationConfig
from learning.observation.robot_state_observation import RobotStateObservationConfig
from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from learning.reinforcement.sb3_policy import HybridLocalGlobalFeaturesExtractor
from paper_training.cpp_env import PaperCPPDiscreteEnv, PaperCPPDiscreteGymEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM + SGTV baseline for online CPP.")
    parser.add_argument("--train-maps", nargs="+", type=Path, default=[REPO_ROOT / "maps" / "indoor_seed101.txt"])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--max-episode-steps", type=int, default=1_500)
    parser.add_argument("--sensor-range", type=int, default=3)
    parser.add_argument("--local-crop-size", type=int, default=41)
    parser.add_argument("--global-sizes", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--save-path", type=Path, default=REPO_ROOT / "artifacts" / "lstm_sgtv.zip")
    parser.add_argument("--no-action-shield", action="store_true")

    # Only the fixed SGTV reward family is exposed in this study runner.
    parser.add_argument("--sgtv-scale", type=float, default=0.1)
    parser.add_argument("--sgtv-horizon", type=int, default=32)
    parser.add_argument("--sgtv-gate-horizon", type=int, default=20)
    parser.add_argument("--sgtv-min-new-cells", type=int, default=1)
    return parser.parse_args()


def load_grid(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    grid = np.asarray(rows, dtype=np.int32)
    if grid.ndim != 2 or grid.size == 0 or not np.isin(grid, [0, 1]).all():
        raise ValueError(f"{path} must be a non-empty 0/1 grid map")
    return grid


def first_free_cell(grid: np.ndarray) -> tuple[int, int]:
    cells = np.argwhere(grid == 0)
    if len(cells) == 0:
        raise ValueError("map has no free cell")
    return tuple(int(x) for x in cells[0])


def make_configs(args: argparse.Namespace, robot_state_dim: int, local_channels: int, global_channels: Sequence[int]):
    reward = CPPRewardConfig(
        new_cell_reward=0.7,
        local_tv_reward_scale=1.0,
        local_tv_normalizer=2.5,
        semi_global_tv_reward_scale=float(args.sgtv_scale),
        semi_global_tv_horizon=int(args.sgtv_horizon),
        semi_global_tv_gate_horizon=int(args.sgtv_gate_horizon),
        semi_global_tv_min_new_cells=int(args.sgtv_min_new_cells),
        semi_global_tv_window_size=int(args.local_crop_size),
        collision_reward=-10.0,
        constant_reward=-0.1,
        constant_reward_always=True,
        turn_change_penalty=-0.05,
        revisit_penalty=-0.2,
    )
    env = CPPDiscreteEnvConfig(
        sensor_range=int(args.sensor_range),
        max_steps=int(args.max_episode_steps),
        stop_on_full_coverage=True,
        include_dtm=False,
        maps_observation_mode="hybrid_local_global",
        use_action_mask=not bool(args.no_action_shield),
        action_mask_mode="shield",
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=int(args.local_crop_size),
            global_coarse_sizes=tuple(int(x) for x in args.global_sizes),
            unknown_policy="keep",
        ),
        robot_state=RobotStateObservationConfig(
            include_position=False,
            include_action_history=True,
            include_action_mask=True,
            include_progress=False,
            include_stagnation=False,
        ),
        reward=reward,
    )
    encoder = HybridLocalGlobalEncoderConfig(
        local_in_channels=int(local_channels),
        global_in_channels=tuple(int(x) for x in global_channels),
        global_sizes=tuple(int(x) for x in args.global_sizes),
        conv_channels=(32, 64),
        local_embed_dim=256,
        global_embed_dim=128,
        global_encoder_mode="sgcnn",
        sgcnn_target_hw=(16, 16),
        sgcnn_pool_hw=(1, 1),
        robot_state=RobotStateEncoderConfig(input_dim=int(robot_state_dim), hidden_dims=(256, 256)),
        fusion_hidden_dims=(512, 512),
    )
    return env, encoder


def main() -> None:
    args = _parse_args()
    if args.num_envs < 1 or args.n_steps <= 0 or args.batch_size <= 0:
        raise ValueError("--num-envs, --n-steps, and --batch-size must be positive")
    if args.lstm_hidden_size <= 0:
        raise ValueError("--lstm-hidden-size must be positive")

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before training.") from exc

    grids = [load_grid(path) for path in args.train_maps]
    if len({grid.shape for grid in grids}) != 1:
        raise ValueError("all training maps must have the same shape")
    start = first_free_cell(grids[0])

    # Probe once: the environment determines exact channel/state dimensions.
    probe_env_cfg = CPPDiscreteEnvConfig(
        sensor_range=int(args.sensor_range),
        max_steps=int(args.max_episode_steps),
        include_dtm=False,
        maps_observation_mode="hybrid_local_global",
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=int(args.local_crop_size),
            global_coarse_sizes=tuple(int(x) for x in args.global_sizes),
            unknown_policy="keep",
        ),
        robot_state=RobotStateObservationConfig(
            include_position=False,
            include_action_history=True,
            include_action_mask=True,
            include_progress=False,
            include_stagnation=False,
        ),
    )
    probe = PaperCPPDiscreteEnv(grid_map=grids[0], start_pos=start, config=probe_env_cfg)
    raw = probe.reset()
    hybrid = raw["hybrid_maps"]
    local_channels = int(np.asarray(hybrid["local"]).shape[0])
    global_channels = [int(np.asarray(hybrid[f"global_{size}"]).shape[0]) for size in args.global_sizes]
    robot_state_dim = int(np.asarray(raw["robot_state"]).shape[0])
    env_cfg, encoder_cfg = make_configs(args, robot_state_dim, local_channels, global_channels)

    env_kwargs = dict(
        grid_map=grids[0],
        start_pos=start,
        config=env_cfg,
        grid_map_pool=grids,
        episode_map_refresh=len(grids) > 1,
        map_refresh_mode="cycle",
        map_refresh_seed=int(args.seed),
    )
    vec_cls = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    vec_env = make_vec_env(
        PaperCPPDiscreteGymEnv,
        n_envs=int(args.num_envs),
        seed=int(args.seed),
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
    )
    policy_kwargs = dict(
        features_extractor_class=HybridLocalGlobalFeaturesExtractor,
        features_extractor_kwargs=dict(encoder_config=encoder_cfg),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        lstm_hidden_size=int(args.lstm_hidden_size),
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        vec_env,
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        policy_kwargs=policy_kwargs,
        seed=int(args.seed),
        device=args.device,
        verbose=1,
    )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "condition": "lstm_sgtv",
        "dtm": False,
        "heuristic_assist": False,
        "train_maps": [str(path) for path in args.train_maps],
    }
    args.save_path.with_suffix(".json").write_text(json.dumps(run_config, indent=2, default=str))
    model.learn(total_timesteps=int(args.total_timesteps))
    model.save(args.save_path)
    vec_env.close()


if __name__ == "__main__":
    main()
