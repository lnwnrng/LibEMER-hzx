## LibEMER:A NOVEL BENCHMARK AND ALGORITHMS LIBRARY FOR EEG-BASED MULTIMODAL EMOTION RECOGNITION

LibEMER 是一个面向 EEG-based multimodal emotion recognition 的 benchmark 与算法库，覆盖 SEED、SEED-V 和 DEAP 等数据集，支持 EEG 与外周生理信号的融合建模，并提供 subject-dependent (SD) 与 subject-independent (SI) 两类常用评测协议。

LibEMER is a benchmark and algorithms library for EEG-based multimodal emotion recognition. It supports multimodal fusion of EEG and peripheral physiological signals, covers datasets such as SEED, SEED-V, and DEAP, and provides both subject-dependent (SD) and subject-independent (SI) evaluation protocols.

### Example Commands / 示例命令

Run from the `LibEMER/` directory.

```bash
python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot


python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used valence -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```
