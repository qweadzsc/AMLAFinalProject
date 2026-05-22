# LC/POMO 路线可执行步骤

## 总体路线

本项目建议选择 `Project/baselines/lc_baseline/` 作为主线，在 POMO baseline 上做轻量、可验证、适合写报告的改进：

1. 先建立 baseline 结果和可复现实验脚本。
2. 复现 Leader Reward 思路，改进 POMO 的训练目标。
3. 加入 Sym-NCO 风格的几何对称增强，提升泛化。
4. 针对跨规模测试，加入 TSP-50/TSP-100 的轻量兼容与评估。
5. 做消融实验并整理 `README.md`、`Report.pdf`。

核心原则：每一步都要有可以用代码验证的 milestone，避免只停留在论文描述。

## Step 1：跑通原始 LC baseline（已完成，NV 环境验证通过）

当前状态：

- 已在 Linux/NVIDIA A800 环境中新建 `amla_tsp` conda 环境，Python 版本为 3.8。
- 已安装 `Project/requirements.txt` 中依赖，包括 `torch==2.1.0`、`ml4co-kit==0.3.3`、`torch-scatter`、`torch-sparse`、`torch-spline-conv`、`torch-cluster`。
- `ml4co-kit` 首次导入时已完成本地 C/C++ 扩展编译，`torch.cuda.is_available()` 返回 `True`。
- 已用短训练配置跑通 `train_lc.py`：`AMLA_EPOCHS=1`、`AMLA_BATCHES_PER_EPOCH=2`、`AMLA_BATCH_SIZE=4`、`AMLA_DEVICE=cuda:0`。
- 训练已生成 `Project/baselines/lc_baseline/checkpoints/best_model.pth`，并生成时间戳目录 `checkpoints/20260522_143953/`。
- 已跑通 `evaluate_lc.py`，评测输出包含 `Average cost`、`Average gap`、`Total time`。

本次 Step 1 验证结果：

```text
Train Length: 24.3000
Train Loss: -0.9764
Val Cost: 12.8206
Average cost: 12.8206
Average optimal: 5.6709
Average gap: 126.30%
Total time: 2.93s
Avg time/instance: 0.0229s
```

目标：确认当前项目环境、LC 模型、训练脚本和评测脚本都能正常运行。

需要完成：

- 进入 `Project/baselines/lc_baseline/`
- 跑一次短训练，确认 `train_lc.py` 可以完成至少 1 个 epoch
- 跑一次 `evaluate_lc.py`，确认可以加载 `checkpoints/best_model.pth` 并输出 cost/gap/time

建议先把训练配置临时改小：

```python
EPOCHS = 1
BATCHES_PER_EPOCH = 2
BATCH_SIZE = 4
```

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py
python evaluate_lc.py
```

验证标准：

- 训练脚本能生成 checkpoint 目录
- 至少保存出一个 `best_model.pth`
- 评测脚本输出 `Average cost`、`Average gap`、`Total time`

如果这一步失败，后续所有实验都没有可靠基础。

## Step 2：建立三类验证集评估入口（已完成）

当前状态：

- 未修改 `Project/baselines/` 和 `Project/data/` 中的原始文件。
- 已新增独立评估目录：`Project/experiments/lc_eval/`。
- `evaluate_lc_dataset.py` 支持通过命令行传入 `--test-data`、`--node-cnt`、`--pomo-size`、`--checkpoint`、`--device`。
- `eval_all_lc.py` 会依次评估三类验证集，并把 JSON 结果写入 `Project/experiments/lc_eval/results/`。

本次 Step 2 验证结果：

```text
tsp50_uniform_val_128:
  Average cost: 12.8206
  Average optimal: 5.6709
  Average gap: 126.30%
  Total time: 2.94s

tsp50_ood_val_16:
  Average cost: 9.9655
  Average optimal: 4.8343
  Average gap: 105.72%
  Total time: 0.55s

tsp100_uniform_val_16:
  Average cost: 21.9777
  Average optimal: 7.8196
  Average gap: 181.34%
  Total time: 0.92s
```

目标：把作业要求的三类验证集都纳入固定评测流程。

需要完成：

- 让 `evaluate_lc.py` 可以方便切换以下数据：
  - `../../data/val/tsp50_uniform_val_128.txt`
  - `../../data/val/tsp50_ood_val_16.txt`
  - `../../data/val/tsp100_uniform_val_16.txt`
- 对 TSP-100 评测时，确保 `NODE_CNT = 100`、`POMO_SIZE = 100`
- 记录每个验证集的 cost、gap、time

推荐做法：

- 在 `evaluate_lc.py` 中增加命令行参数，如 `--test-data`、`--node-cnt`、`--pomo-size`
- 或者新增一个小脚本 `run_eval_all.sh`/`eval_all.py` 批量调用三次评测

Milestone：

```bash
cd Project/baselines/lc_baseline
python evaluate_lc.py --test-data ../../data/val/tsp50_uniform_val_128.txt --node-cnt 50 --pomo-size 50
python evaluate_lc.py --test-data ../../data/val/tsp50_ood_val_16.txt --node-cnt 50 --pomo-size 50
python evaluate_lc.py --test-data ../../data/val/tsp100_uniform_val_16.txt --node-cnt 100 --pomo-size 100
```

验证标准：

- 三个命令都能独立跑完
- 每个命令都输出 `Average cost`、`Average gap`、`Total time`
- TSP-100 不因为 shape、mask、POMO size 报错

这一步的结果就是报告中的 baseline 表格。

## Step 3：实现 Leader Reward 训练目标（已完成）

当前状态：

- 未修改 `Project/baselines/` 和 `Project/data/` 中的原始文件。
- 已新增独立训练入口：`Project/experiments/lc_leader/train_lc_leader.py`。
- 依据 Leader Reward 论文，将同一问题 POMO rollout 中 reward 最大的 leader 轨迹 advantage 乘以 `--leader-reward-multiplier`，并保留 `--normalize-leader-advantage` 选项。
- `--use-leader-reward 0` 时直接使用原始 POMO loss，训练路径退化为 baseline。
- 脚本会打印 `POMO Loss`、`Leader Delta Loss`、`Loss`，并把 checkpoint 保存到 `Project/experiments/lc_leader/checkpoints/<run-name>/`，不覆盖 baseline checkpoint。

本次 Step 3 验证结果：

```text
Leader Reward smoke run:
  command: conda run -n amla_tsp python Project/experiments/lc_leader/train_lc_leader.py --run-name leader_smoke --epochs 1 --batches-per-epoch 2 --batch-size 4 --use-leader-reward 1 --leader-reward-multiplier 2.0 --device cuda:0
  Train Length: 20.9161
  Best Length: 15.7775
  POMO Loss: -17.5356
  Leader Delta Loss: 14.0245
  Loss: -3.5111
  Val Cost: 9.2306

Baseline switch smoke run:
  command: conda run -n amla_tsp python Project/experiments/lc_leader/train_lc_leader.py --run-name baseline_switch_smoke --epochs 1 --batches-per-epoch 1 --batch-size 4 --use-leader-reward 0 --device cuda:0
  POMO Loss: -14.6136
  Leader Delta Loss: 0.0000
  Loss: -14.6136
  Val Cost: 11.0196
```

目标：复现 Leader Reward 的核心思想，让训练目标更贴近 POMO 推理阶段“取最优 rollout”的行为。

当前 baseline 在 `train_lc.py` 中使用：

```python
advantage = reward - reward.mean(dim=1, keepdim=True)
loss = -(advantage * log_prob).mean()
```

需要完成：

- 在 `train_one_batch` 中保留原始 POMO loss
- 额外计算每个 batch 中 reward 最大的 leader rollout
- 给 leader rollout 增加额外强化项
- 增加可配置超参数，例如：
  - `USE_LEADER_REWARD = True`
  - `LEADER_REWARD_WEIGHT = 1.0`

一种可执行的简单版本：

```python
advantage = reward - reward.mean(dim=1, keepdim=True)
pomo_loss = -(advantage * log_prob).mean()

leader_idx = reward.argmax(dim=1)
leader_log_prob = log_prob[torch.arange(batch_size), leader_idx]
leader_advantage = reward.max(dim=1).values - reward.mean(dim=1)
leader_loss = -(leader_advantage.detach() * leader_log_prob).mean()

loss = pomo_loss + LEADER_REWARD_WEIGHT * leader_loss
```

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py
```

验证标准：

- 训练不会出现 NaN
- 日志中能同时打印 `pomo_loss`、`leader_loss`、`loss`
- 关闭 `USE_LEADER_REWARD` 时，训练行为退化为原始 baseline
- 开启 `USE_LEADER_REWARD` 时，至少可以完成 1 个 epoch 并保存模型

报告中可以把这一项作为主要论文复现点。

## Step 4：做 Leader Reward 消融实验

目标：验证 Leader Reward 是否真的带来改进，或者至少得到可分析的实验现象。

需要完成：

- 训练 baseline：`USE_LEADER_REWARD = False`
- 训练 leader 版本：`USE_LEADER_REWARD = True`
- 其他超参数尽量保持一致
- 每个版本都在三个验证集上评测

建议先做小规模快速实验：

```python
EPOCHS = 5
BATCHES_PER_EPOCH = 20
BATCH_SIZE = 64
```

如果时间和算力允许，再扩大到更长训练。

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py --use-leader-reward 0 --run-name baseline_short
python train_lc.py --use-leader-reward 1 --leader-reward-weight 1.0 --run-name leader_short
python evaluate_lc.py --checkpoint checkpoints/baseline_short/best_model.pth --test-data ../../data/val/tsp50_uniform_val_128.txt --node-cnt 50 --pomo-size 50
python evaluate_lc.py --checkpoint checkpoints/leader_short/best_model.pth --test-data ../../data/val/tsp50_uniform_val_128.txt --node-cnt 50 --pomo-size 50
```

验证标准：

- 两个 run 都能生成各自的 `best_model.pth`
- 两个模型都能在同一个验证集上评测
- 能得到一张包含 cost/gap/time 的对比表

即使 leader 版本没有明显超过 baseline，也可以在报告中分析训练长度、超参数和方差问题。

## Step 5：加入几何对称增强

目标：利用 TSP 的几何对称性，提高 OOD 和跨规模泛化。

可实现的增强包括：

- 随机交换 x/y 坐标
- 随机做 `x -> 1 - x`
- 随机做 `y -> 1 - y`
- 可选：90/180/270 度旋转，等价于坐标翻转与交换组合

建议新增函数：

```python
def augment_coordinates(coords):
    ...
    return coords
```

落点可以是：

- `TSPEnv.load_problems` 生成坐标后立即增强
- 或 `train_one_batch` 中 `env.load_problems(batch_size)` 后增强 `env.coordinates` 并重算 `env.problems`

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py --use-aug 1 --epochs 1 --batches-per-epoch 2
```

验证标准：

- 增强后的 `coordinates` 仍在 `[0, 1]`
- `problems = torch.cdist(coordinates, coordinates, p=2)` 被正确重算
- 训练能正常反向传播，不出现 shape 错误
- 同一批数据增强前后 tour 长度量级合理

这一步可以作为第二个改进点，报告中对应 Sym-NCO 风格的对称性利用。

## Step 6：做增强策略消融

目标：确认几何增强是否改善 OOD 或 TSP-100 表现。

需要完成四组对比：

| 实验 | Leader Reward | 几何增强 |
| --- | --- | --- |
| A | 关闭 | 关闭 |
| B | 开启 | 关闭 |
| C | 关闭 | 开启 |
| D | 开启 | 开启 |

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py --use-leader-reward 0 --use-aug 0 --run-name ablation_A
python train_lc.py --use-leader-reward 1 --use-aug 0 --run-name ablation_B
python train_lc.py --use-leader-reward 0 --use-aug 1 --run-name ablation_C
python train_lc.py --use-leader-reward 1 --use-aug 1 --run-name ablation_D
```

验证标准：

- 四个 run 均能保存 `best_model.pth`
- 每个 run 至少在 `tsp50_uniform_val_128` 和 `tsp50_ood_val_16` 上完成评测
- 形成一张消融表，列出 `avg cost`、`avg gap`、`total time`

如果算力不足，可以减少 epoch，但必须保证四组实验训练预算一致。

## Step 7：处理 TSP-100 跨规模评测

目标：确保最终模型可以在 TSP-100 上被外部脚本或本地脚本评测。

需要完成：

- 检查 `LCModel` 是否依赖固定节点数
- 检查 `TSPEnv` 是否能在 `node_cnt=100, pomo_size=100` 下正常构造 mask
- 检查 `evaluate_lc.py` 是否能通过参数创建 TSP-100 环境
- 注意权重来自 TSP-50 训练，但模型结构本身应能接受不同节点数

Milestone：

```bash
cd Project/baselines/lc_baseline
python evaluate_lc.py --checkpoint checkpoints/best_model.pth --test-data ../../data/val/tsp100_uniform_val_16.txt --node-cnt 100 --pomo-size 100
```

验证标准：

- 评测能跑完
- 输出 TSP-100 的 cost/gap/time
- 不出现 attention shape、mask shape、POMO index 越界等错误

如果 TSP-100 表现很差，也可以作为报告中的泛化限制分析。

## Step 8：可选实现混合规模训练

目标：进一步改善 TSP-100 泛化。

可选方案：

- 训练时以一定概率使用 `node_cnt=100, pomo_size=100`
- 或在每个 epoch 中混入少量 TSP-100 batch
- 注意 batch size 可能需要降低，避免显存不足

建议配置：

```python
MIXED_SCALE_TRAINING = True
TSP100_PROB = 0.2
TSP100_BATCH_SIZE = 16
```

Milestone：

```bash
cd Project/baselines/lc_baseline
python train_lc.py --mixed-scale 1 --epochs 1 --batches-per-epoch 2
```

验证标准：

- 训练日志能显示当前 batch 使用的是 TSP-50 还是 TSP-100
- TSP-50 batch 和 TSP-100 batch 都能完成 forward/backward
- 最终 checkpoint 能分别在 TSP-50 和 TSP-100 验证集上评测

这是增强项，不是必须项；如果时间不足，优先保证 Step 3 到 Step 7。

## Step 9：保存最终提交权重

目标：让外部评测脚本能按作业要求自动加载模型。

需要完成：

- 选择验证表现最好的 run
- 将其权重复制或保存为：

```text
Project/baselines/lc_baseline/checkpoints/best_model.pth
```

- 确保 `evaluate_lc.py` 中的 `model_params` 和 `env_params` 与最终模型兼容
- 如果新增了模型参数，必须写进 `model_params`

Milestone：

```bash
cd Project/baselines/lc_baseline
python evaluate_lc.py --checkpoint checkpoints/best_model.pth --test-data ../../data/val/tsp50_uniform_val_128.txt --node-cnt 50 --pomo-size 50
```

验证标准：

- 不指定特殊训练目录时，默认 `checkpoints/best_model.pth` 可被加载
- `model.load_state_dict(...)` 不报 missing key 或 unexpected key
- 默认评测命令可以输出结果

这是最终提交前必须通过的检查。

## Step 10：整理 README 和技术报告

目标：把代码改动和实验现象转化为可评分材料。

`README.md` 建议包含：

- 选择 LC/POMO 的原因
- 主要改动：
  - Leader Reward
  - 几何对称增强
  - 可选混合规模训练
- 训练命令
- 评测命令
- 最终权重位置

`Report.pdf` 建议包含：

- 方法概述
- Leader Reward 复现说明
- 几何增强设计
- 实验设置
- 三类验证集结果表
- 消融实验表
- 失败尝试和限制分析

Milestone：

```bash
cd Project/baselines/lc_baseline
python evaluate_lc.py --test-data ../../data/val/tsp50_uniform_val_128.txt --node-cnt 50 --pomo-size 50
python evaluate_lc.py --test-data ../../data/val/tsp50_ood_val_16.txt --node-cnt 50 --pomo-size 50
python evaluate_lc.py --test-data ../../data/val/tsp100_uniform_val_16.txt --node-cnt 100 --pomo-size 100
```

验证标准：

- 报告中的表格数字都能由命令复现
- README 中的命令能直接运行
- 最终提交目录满足作业接口要求

## 最小可交付版本

如果时间紧，至少完成：

1. Step 1：跑通 LC baseline。
2. Step 2：建立三类验证集评估。
3. Step 3：实现 Leader Reward。
4. Step 4：完成 baseline vs Leader Reward 消融。
5. Step 9：保存最终 `checkpoints/best_model.pth`。
6. Step 10：写清楚 README 和 Report。

这已经足够形成一个完整的课程项目：有论文动机、有代码实现、有实验对比、有报告分析。

## 推荐最终实验表

| 方法 | TSP-50 Uniform Gap | TSP-50 OOD Gap | TSP-100 Gap | Time |
| --- | --- | --- | --- | --- |
| LC baseline | 待填 | 待填 | 待填 | 待填 |
| + Leader Reward | 待填 | 待填 | 待填 | 待填 |
| + Augmentation | 待填 | 待填 | 待填 | 待填 |
| + Leader Reward + Augmentation | 待填 | 待填 | 待填 | 待填 |

报告中优先解释趋势，不要只追求单次最好数字。训练时间短时结果可能波动，应明确说明实验预算。
