"""CPP environment and reward primitives used by both memory conditions."""

from .cpp_env import ACTION_TO_DELTA, CPPDiscreteEnv, CPPDiscreteEnvConfig
from .reward import (
    CPPRewardBreakdown,
    CPPRewardConfig,
    CPPRewardInput,
    compute_cpp_reward,
    total_variation,
)

__all__ = [
    "CPPDiscreteEnv",
    "CPPDiscreteEnvConfig",
    "ACTION_TO_DELTA",
    "CPPRewardConfig",
    "CPPRewardInput",
    "CPPRewardBreakdown",
    "compute_cpp_reward",
    "total_variation",
]
