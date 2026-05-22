# AMLA 大作业任务说明

## 作业主题

本次课程项目要求使用深度学习方法求解旅行商问题（Traveling Salesman Problem, TSP），重点关注训练阶段的 **TSP-50**：

- 输入：50 个城市的二维坐标，城市坐标位于 `[0, 1]^2`
- 目标：构造一条从某个城市出发、访问每个城市恰好一次并回到起点的最短哈密顿回路
- 核心指标：预测路径长度相对于 Concorde 最优解的 **Optimality Gap (%)**

项目目标不是从零实现全部代码，而是在 `Project/baselines/` 中给定的 baseline 基础上选择一种范式进行改进，并在求解质量、推理效率和泛化能力之间取得更好表现。

## 可选技术路线

需要在以下两种范式中选择一种作为主要改进对象。

### 1. GP：Global Prediction

路径位置：`Project/baselines/gp_baseline/`

GP 是基于监督学习的非自回归全局预测方法：

- 模型输入 TSP 图的节点坐标和边距离信息
- GNN 预测每条边是否属于优质或最优 tour
- 推理阶段根据边概率 heatmap 使用贪心解码构造合法回路
- 训练主要使用边标签上的交叉熵损失

可改进方向包括：

- GNN 层数、隐藏维度、聚合方式等超参数
- 边特征、距离特征、拓扑特征、位置编码
- encoder / decoder 结构
- loss 设计、数据增强、训练策略
- 解码策略与后处理策略

### 2. LC：Local Construction

路径：`Project/baselines/lc_baseline/`

LC 是基于强化学习的自回归局部构造方法：

- 将 TSP 构造过程建模为序列决策问题
- 模型逐步选择下一个访问城市
- 使用 Transformer encoder + attention decoder
- 使用 POMO 风格的多起点 rollout 和 REINFORCE 训练

可改进方向包括：

- embedding 维度、attention 层数、head 数等超参数
- decoder 的注意力打分和距离信息融合
- POMO rollout、baseline、采样策略
- 训练数据生成方式、课程学习、数据增强
- 跨分布、跨规模泛化设计

## 数据与验证集

数据位于 `Project/data/`：

- `Project/data/train/tsp50_uniform_train_128k.txt`
  - 128k 个 TSP-50 uniform 训练实例
- `Project/data/val/tsp50_uniform_val_128.txt`
  - 128 个 TSP-50 uniform 验证实例
- `Project/data/val/tsp50_ood_val_16.txt`
  - 16 个 TSP-50 未知分布验证实例，用于 OOD 泛化测试
- `Project/data/val/tsp100_uniform_val_16.txt`
  - 16 个 TSP-100 uniform 验证实例，用于跨规模泛化测试

最终评测使用隐藏测试集，样例验证集只用于本地自测和调参参考。

## 评测指标

最终评测包含三类测试场景：

| 测试集 | 内容 | 权重 |
| --- | --- | --- |
| In-distribution | TSP-50 Uniform | 60% |
| Cross-distribution | TSP-50 未见分布 | 20% |
| Cross-scale | TSP-100 Uniform | 20% |

主要指标：

```text
Gap = (L(predicted_tour) - L(optimal_tour)) / L(optimal_tour) * 100%
```

其中 `L(tour)` 表示 tour 总长度。Gap 越低越好。

次要指标：

- solving time，即所有测试实例的总推理时间
- 推理效率会作为实际部署场景下的重要参考

评分会综合考虑：

- 相对于所选范式 baseline 的提升
- 求解精度
- 推理效率
- OOD 与跨规模泛化
- 方法合理性与创新性
- 技术报告质量

## 必须保持的接口

外部评测脚本会自动导入模型并加载权重，因此接口不能破坏。

### GP 接口要求

如果选择 GP，必须保证：

- `Project/baselines/gp_baseline/model/gp_model.py` 中存在 `GPModel`
- `GPModel.forward(coords)` 返回边 logits，形状为 `(B, V, V, 2)`
- `GPModel.solve(coords)` 返回长度为 batch size 的 tour 列表
- 每个 tour 形状为 `(V + 1,)`
- 每个 tour 必须从节点 `0` 开始，并以节点 `0` 结束
- `Project/baselines/gp_baseline/model/__init__.py` 必须导出 `GPModel`
- `Project/baselines/gp_baseline/evaluate_gp.py` 必须定义全局变量 `model_params`

最终权重必须放在：

```text
Project/baselines/gp_baseline/checkpoints/best_model.pth
```

### LC 接口要求

如果选择 LC，必须保证：

- `Project/baselines/lc_baseline/model/lc_model.py` 中存在 `LCModel`
- `LCModel.__init__` 参数接口保持兼容
- `LCModel.pre_forward(reset_state)` 可接收包含 `coords` 和 `dist` 的 reset state
- `LCModel.forward(state)` 返回 `(selected, prob)`
- `Project/baselines/lc_baseline/model/tsp_env.py` 中存在 `TSPEnv`
- `TSPEnv` 必须支持 `load_problems`、`load_problems_manual`、`reset`、`pre_step`、`step`
- `Project/baselines/lc_baseline/model/__init__.py` 必须导出 `LCModel` 和 `TSPEnv`
- `Project/baselines/lc_baseline/evaluate_lc.py` 必须定义全局变量 `model_params` 和 `env_params`

最终权重必须放在：

```text
Project/baselines/lc_baseline/checkpoints/best_model.pth
```

## 运行 baseline

### GP baseline

```bash
cd Project/baselines/gp_baseline
python train_gp.py
python evaluate_gp.py
```

训练配置主要在 `train_gp.py` 顶部修改。当前 baseline 使用：

- `NUM_LAYERS = 6`
- `HIDDEN_DIM = 128`
- `AGGREGATION = 'sum'`
- `BATCH_SIZE = 64`
- `EPOCHS = 100`
- `LEARNING_RATE = 2e-4`

### LC baseline

```bash
cd Project/baselines/lc_baseline
python train_lc.py
python evaluate_lc.py
```

训练配置主要在 `train_lc.py` 顶部修改。当前 baseline 使用：

- `NODE_CNT = 50`
- `POMO_SIZE = 50`
- `EMBEDDING_DIM = 128`
- `NUM_ATT_LAYERS = 3`
- `NUM_HEADS = 8`
- `QKV_DIM = 16`
- `FF_HIDDEN_DIM = 512`
- `BATCH_SIZE = 64`
- `EPOCHS = 100`
- `LEARNING_RATE = 1e-4`

## 建议工作流程

1. 阅读 `Project/INSTRUCTION.md` 和所选 baseline 代码。
2. 选择 GP 或 LC 作为主要路线。
3. 先跑通对应的 `train_*.py` 和 `evaluate_*.py`。
4. 记录 baseline 在三个样例验证集上的 cost、gap 和推理时间。
5. 设计一到多个改进点，并保持外部评测接口不变。
6. 每次改动后用验证集对比：
   - `tsp50_uniform_val_128.txt`
   - `tsp50_ood_val_16.txt`
   - `tsp100_uniform_val_16.txt`
7. 保存最优权重为对应目录下的 `checkpoints/best_model.pth`。
8. 编写技术报告 `Report.pdf` 和提交目录内的 `README.md`。

## 技术报告要求

最终需要提交技术报告，建议包含：

- 方法概述：选择了 GP 还是 LC，核心改进是什么
- 模型架构：说明相较 baseline 的结构变化
- 训练配置：batch size、learning rate、epochs、hidden dim 等
- 实验结果：验证集上的 cost、gap、time，与 baseline 对比
- 消融实验：如果尝试了多种改进，说明哪些有效、哪些无效
- 设计动机与分析：解释为什么这样改，以及实验现象是否符合预期
- 未来改进方向

失败尝试也可以写入报告，只要分析清楚原因，同样有价值。

## 最终提交内容

选择一种范式提交即可。

### 如果提交 GP

需要保证 `Project/baselines/gp_baseline/` 下包含：

- `model/`
- `utils/`
- `checkpoints/best_model.pth`
- `evaluate_gp.py`
- `train_gp.py`
- `README.md`

### 如果提交 LC

需要保证 `Project/baselines/lc_baseline/` 下包含：

- `model/`
- `checkpoints/best_model.pth`
- `evaluate_lc.py`
- `train_lc.py`
- `README.md`

另外还需要提交技术报告：

```text
Report.pdf
```

## 注意事项

- 只能选择一种主要范式提交，但可以在方法上借鉴另一种范式的思想。
- 可以自由修改 baseline 内部实现，但必须保持指定接口兼容。
- 如果新增模型初始化参数，必须同步写入对应 `evaluate_*.py` 的配置字典。
- 权重文件名必须是 `best_model.pth`。
- 最终隐藏测试集比样例验证集更大，且分布更多样，因此不要只针对样例集过拟合。
- 技术报告的实验记录和分析深度会影响评分。
