# RL CPP: LSTM vs TTT memory

This repository isolates one research question from the CPP project:

> In a partially observable online coverage-path-planning environment, can a
> TTT memory replace the recurrent LSTM memory used by PPO?

Both conditions use the same CPP environment, map observation, encoder, PPO
settings, and SGTV reward.  They differ only at the sequence-memory boundary.

## Experiment order

1. Verify a short **LSTM + SGTV** smoke run.
2. Fix the map splits, evaluation metrics, seeds, and compute budget.
3. Implement a TTT PPO policy with the same encoder and actor/critic heads.
4. Compare LSTM and TTT across the fixed protocol.

## Current layout

- `learning/`, `paper_training/`: minimum CPP environment, observation, SGTV
  reward, and SB3 feature extractors.
- `experiments/train_lstm_sgtv.py`: imported LSTM baseline runner.  It is kept
  temporarily as a behavioural reference while its legacy options are removed.
- `maps/`: five fixed indoor maps for held-out evaluation.
- `cpp_ttt/`: TTT-specific memory and PPO code will live here.

The baseline uses `RecurrentPPO` and `MultiInputLstmPolicy`; automatic A*
rescue and DTM are not part of this comparison.

See [UPSTREAM.md](UPSTREAM.md) for the exact source provenance.
