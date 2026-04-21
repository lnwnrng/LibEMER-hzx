# LibEMER Paper Reproduction Commands

本文件专门用于复现论文中的多模态 benchmark 命令，统一按照协议整理，顺序为 `SI` 在前、`SD` 在后。

- 所有命令都需要在 `LibEMER/` 目录下执行。
- 所有命令都显式带有 `CUDA_VISIBLE_DEVICES=0`，如需切换 GPU，请自行修改前面的设备编号。
- 所有命令均去掉了 `nohup` 和日志重定向，方便直接复制运行。
- `DEAP` 命令中的 `-label_used <DEAP_LABEL>` 请替换成 `valence` 或 `arousal`。
- `G2G` 的命令根据当前仓库中的 benchmark 入口整理，因为训练脚本里没有保留完整的注释版复现命令。
- `HetEmotionNet_train.py` 当前 benchmark 入口只支持 `DEAP` 二分类任务，`SEED` 和 `SEED-V` 组合在当前代码中不支持直接复现。

## Subject-Independent (SI)

### BDDAE

```bash
CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot

CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot

CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-4
```

### BimodalLSTM

```bash
CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -sample_length 6 -bio_length 6 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -sample_length 6 -bio_length 6 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 5 -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### CFDA_CSF

```bash
CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 64 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 5e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4
```

### CMCM

```bash
CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -sample_length 10 -bio_length 10 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -sample_length 10 -bio_length 10 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 10 -bio_length 1280 -stride 1 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### CRNN

```bash
CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### DCCA

```bash
CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### DCCA_AM

```bash
CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### G2G

```bash
CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### HetEmotionNet

```bash
# SEED: unsupported by the current HetEmotionNet_train.py benchmark entry
# SEED-V: unsupported by the current HetEmotionNet_train.py benchmark entry

CUDA_VISIBLE_DEVICES=0 python HetEmotionNet_train.py -model HetEmotionNet -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### MCAF

```bash
CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_independent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_independent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sessions 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_independent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

## Subject-Dependent (SD)

### BDDAE

```bash
CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot

CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 200 -lr 1e-3 -onehot

CUDA_VISIBLE_DEVICES=0 python BDDAE_train.py -model BDDAE -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 200 -lr 1e-3
```

### BimodalLSTM

```bash
CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sample_length 6 -bio_length 6 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sample_length 6 -bio_length 6 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python BimodalLSTM_train.py -model BimodalLSTM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 5 -bio_length 640 -stride 1 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### CFDA_CSF

```bash
CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 5e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CFDA_CSF_train.py -model CFDA_CSF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 64 -epochs 100 -lr 1e-4
```

### CMCM

```bash
CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -sample_length 10 -bio_length 10 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -sample_length 10 -bio_length 10 -stride 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CMCM_train.py -model CMCM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 10 -bio_length 1280 -stride 1 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### CRNN

```bash
CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python CRNN_train.py -model CRNN -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### DCCA

```bash
CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_train.py -model DCCA -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### DCCA_AM

```bash
CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python DCCA_AM_train.py -model DCCA_AM -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### G2G

```bash
CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 1 -bio_stride 1 -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python G2G_train.py -model G2G -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### HetEmotionNet

```bash
# SEED: unsupported by the current HetEmotionNet_train.py benchmark entry
# SEED-V: unsupported by the current HetEmotionNet_train.py benchmark entry

CUDA_VISIBLE_DEVICES=0 python HetEmotionNet_train.py -model HetEmotionNet -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -sample_length 1 -stride 1 -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```

### MCAF

```bash
CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEED_PATH> -dataset seed_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting seedv_multimodal_sub_dependent_train_val_test_setting -dataset_path <SEEDV_PATH> -dataset seedv_de_lds -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot

CUDA_VISIBLE_DEVICES=0 python MCAF_train.py -model MCAF -use_multimodal -metrics acc macro-f1 -metric_choose macro-f1 -setting deap_multimodal_sub_dependent_train_val_test_setting -dataset_path <DEAP_PATH> -dataset deap -time_window 1 -feature_type de_lds -bio_length 128 -bio_stride 128 -bounds 5 5 -label_used <DEAP_LABEL> -seed 2025 -onehot -batch_size 32 -epochs 100 -lr 1e-4
```
