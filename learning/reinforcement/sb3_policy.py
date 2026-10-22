"""SB3 feature extractors shared by the LSTM and TTT policies.

The feature extractor deliberately stops at the encoder output.  Memory is a
separate module so LSTM and TTT receive exactly the same observation features.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from gymnasium import spaces

from learning.common import HybridLocalGlobalEncoder, HybridLocalGlobalEncoderConfig

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception as exc:  # pragma: no cover
    BaseFeaturesExtractor = object  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


class HybridLocalGlobalFeaturesExtractor(BaseFeaturesExtractor):
    """Encode local/global CPP maps and robot state into one shared feature vector."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        encoder_config: Optional[HybridLocalGlobalEncoderConfig] = None,
    ):
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError("stable_baselines3 is required for the CPP feature extractor") from _SB3_IMPORT_ERROR

        required = {"local_map", "robot_state"}
        missing = required.difference(observation_space.spaces)
        if missing:
            raise ValueError(f"Hybrid observation space missing keys: {sorted(missing)}")
        cfg = encoder_config or HybridLocalGlobalEncoderConfig()
        self._global_sizes = tuple(int(size) for size in cfg.global_sizes)
        global_keys = {f"global_map_{size}" for size in self._global_sizes}
        missing_global = global_keys.difference(observation_space.spaces)
        if missing_global:
            raise ValueError(f"Hybrid observation space missing global keys: {sorted(missing_global)}")

        encoder = HybridLocalGlobalEncoder(cfg)
        super().__init__(observation_space, features_dim=encoder.output_dim)
        self.encoder = encoder

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        global_maps = {
            size: observations[f"global_map_{size}"].float()
            for size in self._global_sizes
        }
        return self.encoder(
            observations["local_map"].float(),
            global_maps,
            observations["robot_state"].float(),
        )
