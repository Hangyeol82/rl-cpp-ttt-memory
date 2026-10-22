# Upstream provenance

The initial CPP environment, observation encoder, SGTV reward, and LSTM
baseline were extracted from
[`Hangyeol82/RL-based-Coverage-Path-Planning`](https://github.com/Hangyeol82/RL-based-Coverage-Path-Planning)
at commit `dfc3998c6b7a5e7d9788ff905b027b16fed8b339`.

This repository deliberately excludes prior checkpoints, logs, imitation
learning code, DTM experiments, and A* intervention experiments.  It is the
isolated workspace for the POMDP CPP memory comparison:

`shared observation encoder -> {LSTM, TTT} memory -> PPO actor/critic`

The first reproducibility target is **LSTM + SGTV**.  TTT is added only after
that baseline has a small smoke run and a fixed evaluation protocol.
