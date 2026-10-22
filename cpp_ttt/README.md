# TTT integration boundary

The future TTT policy must preserve the baseline interface:

`shared encoder feature -> sequence memory -> actor/value heads`

The policy needs an explicit per-environment fast-weight state and a rollout
buffer that stores that state at sequence boundaries.  The LSTM baseline is
kept separate so the environment, encoder, PPO hyperparameters, and SGTV
reward stay fixed.
