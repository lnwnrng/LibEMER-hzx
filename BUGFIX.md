# Bug Fix Report

本文档记录了对 LibEMER 源码的所有 Bug 修复，共发现并修正 9 处问题。

---

## Bug 1 — `load_data.py`: 测试循环中误用训练数据变量

**文件**: `LibEMER/data_utils/load_data.py:223`

**问题**: `read_seed_multimodal()` 函数的测试集构建循环中，错误地使用了 `eye_train_data` 而非 `eye_test_data`，导致测试集眼动数据实际来自训练集，造成数据泄露。

```python
# 修复前
eye_trial = eye_train_data[index[0]:index[1]+1]

# 修复后
eye_trial = eye_test_data[index[0]:index[1]+1]
```

---

## Bug 2 — `load_data.py`: 数据集名称拼写错误

**文件**: `LibEMER/data_utils/load_data.py:89`

**问题**: `extract_dataset` 集合中，`"seed_asm_lds"` 被错误拼写为 `"see_und_asm_lds"`，导致该数据集类型无法被正确识别。

```python
# 修复前
"see_und_asm_lds"

# 修复后
"seed_asm_lds"
```

---

## Bug 3 — `preprocess.py`: 滑动窗口起始位置计算错误

**文件**: `LibEMER/data_utils/preprocess.py:229`

**问题**: `time_extraction()` 函数中，滑动窗口的起始位置使用 `i * sample_length` 计算，而正确应使用步长 `step = sample_length - noverlap`，否则当 `noverlap > 0` 时窗口不重叠，与预期行为不符。

```python
# 修复前
start = i * sample_length

# 修复后
start = i * step   # step = sample_length - noverlap
```

---

## Bug 4 — `preprocess.py`: 函数签名缺少参数

**文件**: `LibEMER/data_utils/preprocess.py:457`

**问题**: `power_spectrum_extraction()` 函数定义缺少 `extract_bands` 参数，而调用处（`feature_extraction()` 和 `deap_bio_extraction()`）均传入了该参数，导致运行时报错。

```python
# 修复前
def power_spectrum_extraction(data, sample_rate, time_window, overlap):

# 修复后
def power_spectrum_extraction(data, sample_rate, extract_bands, time_window, overlap):
```

---

## Bug 5 — `split.py`: `zip()` 缺失导致迭代错误

**文件**: `LibEMER/data_utils/split.py:386`

**问题**: `merge_to_part_multimodal()` 的 `cross_session` 分支中，`enumerate()` 调用时传入了三个独立可迭代对象而未用 `zip()` 打包，导致 Python 报 `TypeError`。

```python
# 修复前
for idx1, (sub_eeg, sub_bio, sub_label) in enumerate(eeg_data[i], bio_data[i], label[i]):

# 修复后
for idx1, (sub_eeg, sub_bio, sub_label) in enumerate(zip(eeg_data[i], bio_data[i], label[i])):
```

---

## Bug 6 — `Het.py`: 遗留调试输出

**文件**: `LibEMER/models/Het.py:118`

**问题**: `STDCN_with_GRU.forward()` 中残留了调试用的 `print(x.shape)`，会在每次前向传播时向标准输出打印张量形状，影响日志整洁度。

```python
# 修复前
print(x.shape)
x = self.flatten(x)

# 修复后
x = self.flatten(x)
```

---

## Bug 7 — `Het.py` / `HetEmotionNet.py`: 重复赋值

**文件**: `LibEMER/models/Het.py`, `LibEMER/models/HetEmotionNet.py`

**问题**: `__init__()` 中 `self.flatten = nn.Flatten()` 被连续赋值两次，属于冗余代码。

```python
# 修复前
self.flatten = nn.Flatten()
self.flatten = nn.Flatten()

# 修复后
self.flatten = nn.Flatten()
```

---

## Bug 8 — `G2G_train.py`: 从错误模块导入函数

**文件**: `LibEMER/G2G_train.py:9`

**问题**: `CE_Label_Smooth_Loss` 被错误地从 `utils.utils` 中导入，而该函数实际定义在 `models.G2G` 中（文件后续已正确导入）。错误的导入会导致 `ImportError`。

```python
# 修复前
from utils.utils import state_log, result_log, setup_seed, sub_result_log, CE_Label_Smooth_Loss

# 修复后
from utils.utils import state_log, result_log, setup_seed, sub_result_log
# CE_Label_Smooth_Loss 已在下方从 models.G2G 正确导入
```

---

## Bug 9 — 多个 Trainer 文件: `DataLoader(None)` 崩溃

**文件**: 以下 9 个文件均受影响

- `LibEMER/Trainer/training.py`
- `LibEMER/Trainer/BimodalLSTMTraining.py`
- `LibEMER/Trainer/G2GTraining.py`
- `LibEMER/Trainer/BDDAETraining.py`
- `LibEMER/Trainer/MCAFTraining.py`
- `LibEMER/Trainer/CRNNTraining.py`
- `LibEMER/Trainer/CMCMTraining.py`
- `LibEMER/Trainer/DCCA_AMTraining.py`
- `LibEMER/Trainer/CFDA_CSFTraining.py`

**问题**: 当 `test_sub_label=None`（非受试者独立评估场景）时，`DataLoader(test_sub_label, ...)` 被无条件调用，传入 `None` 作为数据集，导致运行时崩溃。代码后续虽有 `if test_sub_label is not None` 的分支判断，但 DataLoader 已在此之前创建并报错。

```python
# 修复前
test_sub_label_loader = DataLoader(test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4)

# 修复后
test_sub_label_loader = DataLoader(
    test_sub_label, sampler=sampler_test, batch_size=batch_size, num_workers=4
) if test_sub_label is not None else None
```

---

## 变更文件汇总

| 文件 | 修复的 Bug |
|------|-----------|
| `LibEMER/data_utils/load_data.py` | Bug 1, Bug 2 |
| `LibEMER/data_utils/preprocess.py` | Bug 3, Bug 4 |
| `LibEMER/data_utils/split.py` | Bug 5 |
| `LibEMER/models/Het.py` | Bug 6, Bug 7 |
| `LibEMER/models/HetEmotionNet.py` | Bug 7 |
| `LibEMER/G2G_train.py` | Bug 8 |
| `LibEMER/Trainer/training.py` | Bug 9 |
| `LibEMER/Trainer/BimodalLSTMTraining.py` | Bug 9 |
| `LibEMER/Trainer/G2GTraining.py` | Bug 9 |
| `LibEMER/Trainer/BDDAETraining.py` | Bug 9 |
| `LibEMER/Trainer/MCAFTraining.py` | Bug 9 |
| `LibEMER/Trainer/CRNNTraining.py` | Bug 9 |
| `LibEMER/Trainer/CMCMTraining.py` | Bug 9 |
| `LibEMER/Trainer/DCCA_AMTraining.py` | Bug 9 |
| `LibEMER/Trainer/CFDA_CSFTraining.py` | Bug 9 |
