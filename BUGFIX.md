# LibEMER Bug Fix Report

本文档记录 LibEMER 仓库的两轮审查与修复结果。第二轮修复以论文 benchmark 约束为准，聚焦会导致运行失败、数据流错误、论文与代码不一致、或结果统计失真的问题；本轮不做大范围风格重构。

## 第一轮修复回顾

### 问题 1: `load_data.py` 中 SEED 多模态测试眼动数据取错来源
- 涉及文件: `LibEMER/data_utils/load_data.py`
- 触发条件: 调用 `read_seed_multimodal()` 读取 SEED 多模态数据。
- 根因: 构建测试集眼动 trial 时误用了 `eye_train_data`，导致测试集眼动特征来自训练集。
- 修复方式: 将测试切片来源更正为 `eye_test_data`。
- 影响范围: SEED 多模态测试集存在数据泄漏风险。
- 论文一致性: 是。该问题会直接破坏 benchmark 的训练/测试隔离。

### 问题 2: `load_data.py` 中 `seed_asm_lds` 数据集名称拼写错误
- 涉及文件: `LibEMER/data_utils/load_data.py`
- 触发条件: 传入 `seed_asm_lds` 配置时。
- 根因: 数据集名称被误写为 `see_und_asm_lds`，导致分支判断失效。
- 修复方式: 更正为 `seed_asm_lds`。
- 影响范围: 指定该数据集时无法走到正确的数据加载逻辑。
- 论文一致性: 否。属于直接运行错误修复。

### 问题 3: `time_extraction()` 的滑窗起点计算错误
- 涉及文件: `LibEMER/data_utils/preprocess.py`
- 触发条件: 使用带重叠的时间窗特征提取。
- 根因: 窗口起始位置使用 `i * sample_length`，忽略了 `step = sample_length - noverlap`。
- 修复方式: 将起点改为 `i * step`。
- 影响范围: 有重叠窗口时切片位置错误，特征分段与预期不一致。
- 论文一致性: 是。该问题会改变实际时间窗划分。

### 问题 4: `power_spectrum_extraction()` 函数签名缺少参数
- 涉及文件: `LibEMER/data_utils/preprocess.py`
- 触发条件: 调用频域功率谱特征提取时。
- 根因: 函数定义缺少 `extract_bands` 参数，但调用方会传入该参数。
- 修复方式: 将函数签名补齐为包含 `extract_bands`。
- 影响范围: 频域特征提取会直接抛出参数错误。
- 论文一致性: 否。属于运行错误修复。

### 问题 5: `merge_to_part_multimodal()` 中 `enumerate()` 使用方式错误
- 涉及文件: `LibEMER/data_utils/split.py`
- 触发条件: 进入 `cross_session` 多模态合并分支时。
- 根因: `enumerate()` 接收了三个独立可迭代对象，而不是先用 `zip()` 打包。
- 修复方式: 改为 `enumerate(zip(...))`。
- 影响范围: 该分支会直接触发 `TypeError`。
- 论文一致性: 否。属于运行错误修复。

### 问题 6: `Het.py` 中遗留调试输出
- 涉及文件: `LibEMER/models/Het.py`
- 触发条件: 执行 `STDCN_with_GRU.forward()`。
- 根因: 前向过程中遗留 `print(x.shape)` 调试语句。
- 修复方式: 删除调试输出。
- 影响范围: 干扰训练与评估日志，不影响数值结果。
- 论文一致性: 否。属于工程清理。

### 问题 7: `Het.py` 与 `HetEmotionNet.py` 中重复定义 `self.flatten`
- 涉及文件: `LibEMER/models/Het.py`, `LibEMER/models/HetEmotionNet.py`
- 触发条件: 初始化模型时。
- 根因: `self.flatten = nn.Flatten()` 被重复赋值。
- 修复方式: 删除重复定义，保留单一初始化。
- 影响范围: 不影响功能，但会增加代码噪声。
- 论文一致性: 否。属于工程清理。

### 问题 8: `G2G_train.py` 从错误模块导入 `CE_Label_Smooth_Loss`
- 涉及文件: `LibEMER/G2G_train.py`
- 触发条件: 导入训练脚本时。
- 根因: `CE_Label_Smooth_Loss` 被误从 `utils.utils` 导入，但实际定义在 `models.G2G`。
- 修复方式: 删除错误导入，使用 `models.G2G` 中的正确定义。
- 影响范围: 可能导致 `ImportError` 或导入混乱。
- 论文一致性: 否。属于运行错误修复。

### 问题 9: 多个 Trainer 在 `test_sub_label=None` 时仍创建 `DataLoader`
- 涉及文件:
- `LibEMER/Trainer/training.py`
- `LibEMER/Trainer/BimodalLSTMTraining.py`
- `LibEMER/Trainer/G2GTraining.py`
- `LibEMER/Trainer/BDDAETraining.py`
- `LibEMER/Trainer/MCAFTraining.py`
- `LibEMER/Trainer/CRNNTraining.py`
- `LibEMER/Trainer/CMCMTraining.py`
- `LibEMER/Trainer/DCCA_AMTraining.py`
- `LibEMER/Trainer/CFDA_CSFTraining.py`
- 触发条件: 非 `sub_independent` 评估流程。
- 根因: 在未判断 `test_sub_label is not None` 之前就调用 `DataLoader(test_sub_label, ...)`。
- 修复方式: 仅在 `test_sub_label` 存在时构建对应 loader。
- 影响范围: 会直接导致评估阶段崩溃。
- 论文一致性: 否。属于运行错误修复。

## 第二轮修复

### 问题 10: `read_seed_feature()` 通过字符串拼接路径，导致 SEED 特征数据无法加载
- 涉及文件: `LibEMER/data_utils/load_data.py`
- 触发条件: 读取 `seed`, `seed4`, `seed5`, `seed_asm_lds` 等特征型数据集。
- 根因: `dataset_path` 与 `Extracted_Features`、`label.mat` 等路径通过字符串直接相连，在路径末尾没有分隔符时会生成错误路径。
- 修复方式: 统一改用 `os.path.join()` 构建目录、标签文件和各个 `.mat` 文件路径。
- 影响范围: SEED 系列特征读取可能直接失败。
- 论文一致性: 是。该问题会阻断论文基准对应数据的实际使用。

### 问题 11: 多模态 `get_data()` 静默强制 `eog_clean=True`
- 涉及文件: `LibEMER/data_utils/load_data.py`
- 触发条件: 调用多模态数据加载，尤其是用户显式关闭 EOG 清洗时。
- 根因: `multimodal_preprocess()` 调用处无条件把 `eog_clean` 设为 `True`，覆盖了配置对象中的真实设定。
- 修复方式: 改为透传 `setting.eog_clean`。
- 影响范围: 配置与实际执行不一致，容易让结果不可复现。
- 论文一致性: 是。伪迹处理是否启用属于预处理协议的一部分。

### 问题 12: `eog_remove()` 名义上做伪迹清洗，实际上是空操作
- 涉及文件: `LibEMER/data_utils/preprocess.py`
- 触发条件: 开启 `eog_clean` 的 EEG 预处理流程。
- 根因: 原实现只是原样返回数据，没有执行任何 PCA 去伪迹步骤。
- 修复方式: 按 trial 执行启发式 PCA 伪迹缓解，将 EEG 转为 time-major 形式，去掉第 1 主成分后再逆变换；若维度不足或 PCA 失败，则保留原 trial。
- 影响范围: 代码会对外宣称“做了 EOG 清洗”，但结果与未清洗完全一致。
- 论文一致性: 是。论文明确提到使用 PCA 缓解眼动伪迹。

### 问题 13: 通用 Trainer 的 `sub_evaluate()` 调用了错误的四输入接口
- 涉及文件: `LibEMER/Trainer/training.py`
- 触发条件: 使用通用训练器评估 HetEmotionNet 这类双输入多模态模型时。
- 根因: `sub_evaluate()` 残留了域适配模型的四输入调用形式，与 `model(eeg_features, bio_features)` 的实际接口不匹配。
- 修复方式: 改为双输入前向调用，并同步修正标签到损失函数与指标计算的使用方式。
- 影响范围: 主体验证或测试阶段会直接报错，或在 one-hot 标签场景下统计失真。
- 论文一致性: 否。属于评估链路正确性修复。

### 问题 14: 多个 Trainer 的 `evaluate()` 定义与调用参数不一致
- 涉及文件:
- `LibEMER/Trainer/CRNNTraining.py`
- `LibEMER/Trainer/MCAFTraining.py`
- `LibEMER/Trainer/CFDA_CSFTraining.py`
- 触发条件: 训练完成后进入最终测试或主体验证分支时。
- 根因: 调用方传入了额外参数，但 `evaluate()` 定义未接收，导致 `TypeError`。
- 修复方式: 统一为这些 `evaluate()` 增加可选的 `loss_func` / `loss_param` 参数，并在需要时正确累计额外损失项。
- 影响范围: 相关模型在 subject-dependent 最终测试阶段可能直接中断。
- 论文一致性: 否。属于运行错误修复。

### 问题 15: `test_sub_label_loader` 的构建与 `drop_last` 策略不一致，导致最后一批样本被静默丢弃
- 涉及文件:
- `LibEMER/Trainer/training.py`
- `LibEMER/Trainer/G2GTraining.py`
- `LibEMER/Trainer/CRNNTraining.py`
- `LibEMER/Trainer/MCAFTraining.py`
- `LibEMER/Trainer/CFDA_CSFTraining.py`
- `LibEMER/Trainer/DCCA_AMTraining.py`
- 触发条件: `sub_independent` 主体验证流程，且测试样本数不能被 batch size 整除。
- 根因: 主测试 loader 不丢最后一批，但主体标签 loader 仍然使用会丢批次的配置，或沿用训练集的 `drop_last` 标志。
- 修复方式: 仅在 `test_sub_label` 存在时创建 loader，并让其 `drop_last` 与对应测试 loader 保持一致。
- 影响范围: 主体级统计会发生样本错位，最后一批数据被静默漏掉。
- 论文一致性: 是。该问题会直接影响主体级评估结果。

### 问题 16: `DCCATraining.py` 的验证/测试 `drop_last` 误复用训练集标志，且仍残留无保护的 `DataLoader(test_sub_label, ...)`
- 涉及文件: `LibEMER/Trainer/DCCATraining.py`
- 触发条件: 使用 DCCA 训练流程，尤其是 batch 数不能整除或 `test_sub_label` 为空时。
- 根因: `val` / `test` loader 错误复用了训练集的 `drop_last` 策略，同时未对 `test_sub_label` 的创建做空值保护。
- 修复方式: 将 `val`、`test` 分别改用 `val_drop_flag`、`test_drop_flag`，并只在 `test_sub_label` 存在时创建主体标签 loader。
- 影响范围: 验证/测试样本会被错误截断，或在无主体标签时直接崩溃。
- 论文一致性: 是。会影响验证/测试集统计完整性。

### 问题 17: `BDDAETraining.py` 预训练最佳 checkpoint 不保证首次落盘
- 涉及文件: `LibEMER/Trainer/BDDAETraining.py`
- 触发条件: 进入预训练阶段后，在首次 epoch 或没有更优 loss 的情况下继续执行正式训练。
- 根因: `checkpoint-bestpretrainloss` 只有在 `loss_value < best_loss` 时才保存，而 `epoch == 0` 时只是更新了 `best_loss`，未同步保存模型。
- 修复方式: 将逻辑改为“首个可用结果或更优结果都保存”，确保最佳预训练权重总是存在。
- 影响范围: 后续加载最佳预训练 checkpoint 时可能直接失败。
- 论文一致性: 否。属于训练流程稳定性修复。

### 问题 18: `G2G.py` 对 DEAP 常量的硬编码会错误处理 SEED/SEED-V 多模态数据
- 涉及文件: `LibEMER/models/G2G.py`
- 触发条件: 用 G2G 运行 SEED 系列多模态任务，或更换类别数配置时。
- 根因: 模型内部写死了 `32` 个 EEG 节点、`8` 个眼动节点、`160/240` 的切片边界、DEAP 坐标表，以及固定 `2` 类输出。
- 修复方式: 新增按数据集选择的配置逻辑，动态确定 EEG/眼动节点数、展平维度、随机节点顺序长度、坐标表和分类头输出维度；如果输入展平维度与数据集设定不符，则显式报错。
- 影响范围: SEED/SEED-V 会错误沿用 DEAP 假设，出现 EEG 截断、眼动节点错位或类别数不匹配。
- 论文一致性: 是。G2G 在不同 benchmark 数据集上的结构输入应与真实通道配置一致。

### 问题 19: `G2G_train.py` 没有把真实类别数传给模型
- 涉及文件: `LibEMER/G2G_train.py`
- 触发条件: G2G 模型在不同数据集或不同标签任务上训练时。
- 根因: 训练脚本创建模型前没有把 `num_classes` 注入 `args`，模型只能退回到默认的二分类输出头。
- 修复方式: 在 DEAP 和 SEED 分支中都先设置 `args.num_classes = num_classes`，再实例化模型。
- 影响范围: 多类别任务会出现分类头维度错误。
- 论文一致性: 是。输出类别数必须与 benchmark 任务定义一致。

### 问题 20: `HetEmotionNet_train.py` 入口与论文 benchmark 约束不一致
- 涉及文件: `LibEMER/HetEmotionNet_train.py`
- 触发条件: 使用 HetEmotionNet 训练脚本时。
- 根因: 入口未限制数据集与任务范围，也未显式启用论文要求的 TnF 与生理特征提取，同时模型维度被硬编码为 `40, 128, 4, 2`。
- 修复方式: 将入口调整为论文诚实模式，只允许 DEAP 二分类任务；显式设置 `TnF=True` 和 `extract_bio=True`；保留 4 个频带配置；对单步序列维度进行压平；根据处理后的 EEG/PPS 张量动态推导节点数、时域维度、频域维度和类别数。
- 影响范围: 旧实现可能在不支持的数据集上“看似可跑”，但与论文基准设置不一致，且容易因硬编码维度导致后续错误。
- 论文一致性: 是。论文中 HetEmotionNet 仅用于 DEAP 二分类多模态设置。

## 本轮静态验证范围

本轮未执行 Python 运行验证，只做静态修复与交叉检查，原因是当前环境未进行可运行训练链路验证，且本轮范围已明确为静态审查。

已完成的静态检查包括：
- 仓库中不再存在无保护的 `DataLoader(test_sub_label, ...)`。
- `CRNNTraining.py`、`MCAFTraining.py`、`CFDA_CSFTraining.py` 的 `evaluate()` 定义已与调用参数对齐。
- `G2G.py` 中已去除对 `160`、`240`、`32` 和 `return_coordinates_deap()` 的无条件 DEAP-only 依赖。
- 多模态数据加载不再静默强制 `eog_clean=True`。
- `HetEmotionNet_train.py` 已显式限制为 DEAP 二分类，并强制开启 TnF 与生理特征提取。
- `BUGFIX.md` 已重写为正常 UTF-8 文本。

## 备注

- 第二轮中的 PCA 去伪迹实现采用“每个 trial 去掉第 1 主成分”的启发式近似，目标是补齐原代码缺失的实际清洗行为，并与论文中的 PCA 伪迹缓解描述保持工程一致性。
- 本轮优先修复正确性、可复现性和论文一致性问题，没有顺手扩展无关 CLI 配置项或做大规模架构调整。
