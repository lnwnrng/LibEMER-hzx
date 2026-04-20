# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibEMER is a benchmark and algorithms library for **EEG-based multimodal emotion recognition**. It implements multiple deep learning models that fuse EEG with peripheral physiological signals (EOG, EMG, GSR, BVP, RESP, TEMP, ECG).

## Running Training

Each model has a dedicated `*_train.py` entry point in `LibEMER/`. Run from the `LibEMER/` directory:

```bash
cd LibEMER

# Example: CRNN with SEED dataset (subject-dependent, multimodal)
python CRNN_train.py \
  -model CRNN \
  -use_multimodal \
  -setting seed_multimodal_sub_dependent_train_val_test_setting \
  -dataset_path /path/to/SEED \
  -dataset seed_de_lds \
  -batch_size 32 -epochs 100 -lr 0.0001 -onehot

# Example: DEAP dataset, subject-independent, valence classification
python CRNN_train.py \
  -model CRNN -use_multimodal \
  -setting deap_multimodal_sub_independent_train_val_test_setting \
  -dataset_path /path/to/deap \
  -dataset deap \
  -bio_length 128 -bio_stride 128 \
  -bounds 5 5 -label_used valence -onehot \
  -batch_size 32 -epochs 100 -lr 1e-4
```

Key CLI flags:
- `-setting`: use a preset experiment configuration (see `config/setting.py:preset_setting` dict)
- `-experiment_mode`: `sub_dependent` | `sub_independent` | `cross_session`
- `-split_type`: `kfold` | `leave_one_out` | `front_back` | `train_val_test`
- `-use_multimodal`: enable EEG + peripheral physiological signal fusion
- `-device`: `cuda` | `cpu`
- `-metrics`: e.g. `acc macro-f1`; `-metric_choose` selects the best-model criterion

## Architecture

### Entry Points → Trainer → Model

Each `*_train.py` script:
1. Parses args via `utils/args.py:get_args_parser()`
2. Builds a `Setting` object (from `config/setting.py`) — either from a preset or from raw args
3. Calls `data_utils/load_data.py:get_data()` to load and preprocess data
4. Instantiates a model from `models/`
5. Delegates training to the corresponding `Trainer/*Training.py`

### Data Pipeline (`data_utils/`)

- `load_data.py` — dataset readers for SEED, SEED-V, DEAP (raw `.mat`/`.pkl`/`.xml`); routes to `preprocess.py`
- `preprocess.py` — bandpass filtering, EOG artifact removal, windowing, feature extraction (DE/LDS), multimodal preprocessing
- `split.py` — implements `sub_dependent`, `sub_independent`, `cross_session` splits; supports k-fold, leave-one-out, front-back, train/val/test
- `constants/seed.py`, `constants/deap.py` — channel names and 2D grid locations for topology-aware models

### Models (`models/`)

| File | Model | Notes |
|------|-------|-------|
| `CRNN.py` | CRNN | Conv-RNN for multimodal EEG+bio |
| `BDDAE.py` | BDDAE | Bimodal dual-stream autoencoder |
| `BimodalLSTM.py` | BimodalLSTM | Dual LSTM streams |
| `CFDA_CSF.py` | CFDA-CSF | Cross-frequency domain attention |
| `CMCM.py` | CMCM | Cross-modal correlation |
| `DCCA.py` / `DCCA_AM.py` | DCCA / DCCA-AM | Deep canonical correlation |
| `G2G.py` | G2G | Graph-to-graph |
| `HetEmotionNet.py` | HetEmotionNet | Heterogeneous emotion network |
| `MCAF.py` | MCAF | Multi-channel attention fusion |
| `Models.py` | shared base classes |

### Config (`config/setting.py`)

`Setting` is the central config object holding all hyperparameters. `preset_setting` is a dict mapping string names to factory functions — pass one via `-setting` to reproduce published experiments. Datasets supported: `seed`, `seed_de_lds`, `seedv`, `seedv_de_lds`, `deap`.

### Trainer (`Trainer/`)

`training.py` is the generic training loop (used by most models). Model-specific trainers (e.g., `CRNNTraining.py`) handle custom forward signatures or loss functions. Outputs are saved to `-output_dir` (default `./result/`) and logs to `-log_dir` (default `./log/`).

### Utils (`utils/`)

- `args.py` — shared `argparse` parser (imported by all `*_train.py`)
- `metric.py` — `Metric` / `SubMetric` classes tracking acc, macro-F1, etc.
- `store.py` — checkpoint saving, output directory creation
- `utils.py` — `setup_seed`, logging helpers

## Supported Datasets

- **SEED** / **SEED-IV**: 3-class / 4-class emotion, 15 subjects, 3 sessions, EEG + EOG/EMG
- **SEED-V**: 5-class emotion, 16 subjects, 3 sessions
- **DEAP**: valence/arousal/dominance/liking (continuous → binary via `-bounds`), 32 subjects

Dataset paths are passed at runtime via `-dataset_path`; the library does **not** download data.
