# DAR + ELG 路线可执行步骤

## 总体判断

这次建议关闭 Leader Reward 路线，改做一条更稳的 attention 改造路线：

1. 先复用当前 LC/POMO baseline 与现有三验证集评测流程。
2. 优先实现 **DAR (Distance-Aware Attention Reshaping)**，因为它是**推理期改动**、**不增加参数**、**不要求重训即可先验证**。
3. 再实现 **ELG (Ensemble with Transferrable Local Policy)** 的 TSP 版本，因为它是**训练期局部策略 + 全局策略融合**，工程量更大，但和当前 POMO/LC 结构兼容。
4. 最后做 **ELG + DAR** 复合，作为我们自己的工作点。

这条路线的优点是：

- 比 Leader Reward 更贴近 attention 本体，和当前模型结构更匹配。
- DAR 可以先给出一个低风险、可快速验证的结果。
- ELG 的局部策略可以作为后续增强项，不需要一开始就重构整个 baseline。
- 两篇论文都有官方开源代码，可以对照实现细节。

## 两篇论文总结

### 1. DAR：Distance-Aware Attention Reshaping for Enhancing Generalization of Neural Solvers

论文核心思想：

- 作者观察到 attention-based neural solver 在从小规模/单一分布泛化到大规模/异分布实例时，会出现 **attention score dispersion**，即高分候选节点变多，导致 next-node 选择不够集中。
- DAR 的做法是在**推理阶段**直接给原始 attention/logit 加一个基于几何距离的 bias，不增加模型参数。
- 直觉上，当前节点附近的候选点更可能是合理下一步，因此在原模型得分之外，额外加入“距离越近越优先”的启发式。

可执行的简化公式：

```text
original_score = decoder logits before softmax
distance_bias(i, j) =
  -log(d_ij), if j in top-K nearest neighbors of current node i
  -d_ij,     otherwise

reshaped_score = original_score + alpha * distance_bias
prob = softmax(mask(reshaped_score))
```

对我们当前仓库的意义：

- 非常适合当前 `LCModel`，因为它本身就有 decoder logits。
- 可以先只在 `Project/experiments/` 新建独立推理脚本，不改 baseline 原文件。
- 可以直接复用我们已有的三验证集：
  - `tsp50_uniform_val_128`
  - `tsp50_ood_val_16`
  - `tsp100_uniform_val_16`

预期收益：

- 重点看 **OOD** 和 **跨规模 TSP100** 是否改善。
- 训练成本很低，因为第一阶段甚至可以只做 inference-time patch。

### 2. ELG：Towards Generalizable Neural Solvers for Vehicle Routing Problems via Ensemble with Transferrable Local Policy

论文核心思想：

- 全局 policy（如 POMO）擅长从完整图中学全局结构，但跨分布、跨规模时泛化弱。
- 作者引入一个 **local policy**，只在当前节点附近的 K 个邻居上做决策，学习更可迁移的局部拓扑模式。
- 推理时把全局 policy 和 local policy 的 score 相加，形成 ensemble。

论文中的关键结构：

```text
u_global_tilde = u_global + normalized distance penalty
u_ens = u_global_tilde + u_local
pi = softmax(mask(C * tanh(u_ens)))
```

local policy 的关键点：

- 只看当前节点附近 K 个最近邻。
- 局部特征强调相对几何结构，而不是整张图的全局表示。
- 论文里用位置编码保留邻居按距离排序后的顺序信息。

对我们当前仓库的意义：

- 当前 LC baseline 是 POMO 风格 rollout，天然适合加一个局部辅助策略。
- 但完整复现 ELG 比 DAR 重得多，因为它涉及：
  - 新的局部输入构造
  - 局部策略网络
  - 全局/局部联合训练
  - 可能需要两阶段训练

因此这次建议的实现方式不是“照搬 CVRP/TSP 官方完整仓库”，而是做一个 **TSP-only, LC-compatible ELG-lite**：

- 只支持当前课程项目用到的 TSP。
- 不碰 `Project/baselines/` 和 `Project/data/`。
- 在 `Project/experiments/` 里新建完整训练/评测入口。

### 3. 适合作为我们的复合工作点

最自然的组合方式是：

1. **训练期**：用 ELG 的局部策略增强全局 policy，提升 learned policy 的可迁移性。
2. **推理期**：再套一层 DAR，对 decoder logits 做距离感知 reshape。

这样组合的好处是：

- ELG 负责“学到更稳的局部-全局协同”。
- DAR 负责“在推理时抑制 attention dispersion”。
- 两者一个偏训练、一个偏推理，接口上不冲突。

## 官方开源仓库

- DAR 官方代码：<https://github.com/ftwangyang/DAR>
- ELG 官方代码：<https://github.com/gaocrr/ELG>

对应论文页面：

- DAR：<https://arxiv.org/abs/2401.06979>
- ELG：<https://arxiv.org/abs/2308.14104>
- ELG IJCAI 2024 论文 PDF：<https://www.ijcai.org/proceedings/2024/764>

## 上一次尝试里可直接复用的内容

以下内容已经打通，新的 DAR/ELG 路线不需要重做：

- `amla_tsp` conda 环境已经可用，PyTorch + ml4co-kit + CUDA 已验证通过。
- 单卡 GPU 训练流程已经验证，`CUDA_VISIBLE_DEVICES=5` + `cuda:0` 可正常跑。
- 三验证集统一评测脚本已经有：
  - `Project/experiments/lc_eval/evaluate_lc_dataset.py`
  - `Project/experiments/lc_eval/eval_all_lc.py`
- 独立训练脚手架已经有，可直接改造成 ELG 训练器：
  - `Project/experiments/lc_leader/train_lc_leader.py`
- 当前 baseline 与长训练结果已经有，可继续作为比较对象：
  - `Project/baselines/lc_baseline/checkpoints/best_model.pth`
  - `Project/experiments/lc_leader/results/long_baseline_e80_b50_seed20260522_gpu5/`
- “不改 baseline/data，只在 experiments 新建目录工作”的工程约束已经实践过，后续继续沿用。

建议复用方式：

- 复用 `train_lc_leader.py` 的 CLI、seed、val interval、checkpoint、GPU 信息打印逻辑，改造成 `train_lc_elg.py`。
- 复用 `evaluate_lc_dataset.py` 的 dataset loading 与 JSON 输出逻辑，改造成 `evaluate_lc_dar.py` / `eval_all_lc_dar.py`。
- 复用之前的结果目录结构，统一把新实验放到 `Project/experiments/lc_dar_elg/` 下。

## 新路线的目录建议

建议新建：

```text
Project/experiments/lc_dar_elg/
  train_lc_elg.py
  evaluate_lc_dar.py
  eval_all_lc_dar.py
  model/
    local_policy.py
    dar_wrapper.py
    lc_elg_policy.py
  results/
  checkpoints/
```

原则：

- 不修改 `Project/baselines/lc_baseline/evaluate_lc.py`
- 不修改 `Project/baselines/lc_baseline/train_lc.py`
- 不修改 `Project/data/`

## Step 1：复用并固定 baseline 评测（已完成）

目标：

- 把当前 long baseline 作为后续 DAR/ELG 的统一对照组。

需要完成：

- 确认 baseline 使用哪个 checkpoint 作为对照。
- 用现有 `eval_all_lc.py` 再跑一次，固定 baseline JSON 输出位置。

Milestone：

```bash
conda run -n amla_tsp python Project/experiments/lc_eval/eval_all_lc.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0
```

可验证成果：

- 生成 baseline 的三验证集 JSON 结果。
- 后续所有 DAR/ELG 结果都和这组 baseline 对比。

本次实际执行命令：

```bash
conda run -n amla_tsp python Project/experiments/lc_eval/eval_all_lc.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --results-dir Project/experiments/lc_eval/results/step1_baseline_refresh
```

本次实验结果：

| Dataset | Average cost | Average optimal | Average gap | Total time | Avg time / instance |
| --- | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 12.8206 | 5.6709 | 126.30% | 2.91s | 0.0228s |
| tsp50_ood | 9.9655 | 4.8343 | 105.72% | 0.55s | 0.0344s |
| tsp100_uniform | 21.9777 | 7.8196 | 181.34% | 0.90s | 0.0562s |

可验证成果对应结果：

- **生成 baseline 的三验证集 JSON 结果**：已完成，结果写入 `Project/experiments/lc_eval/results/step1_baseline_refresh/`，包含：
  - `tsp50_uniform.json`
  - `tsp50_ood.json`
  - `tsp100_uniform.json`
  - `summary.json`
- **后续所有 DAR/ELG 结果都和这组 baseline 对比**：已满足，这组结果现在可以作为后续所有实验的固定对照组；使用的 checkpoint 为 `Project/baselines/lc_baseline/checkpoints/best_model.pth`，设备为 `cuda:0`。

## Step 2：实现 DAR 的独立推理版（已完成）

目标：

- 在不重训模型的前提下，先验证 DAR 是否能改善 OOD 和 TSP100 泛化。

需要完成：

- 新建 `Project/experiments/lc_dar_elg/model/dar_wrapper.py`
- 在 decoder 输出 softmax 之前，拿到 `score_masked` 或等价 logits
- 根据当前节点到所有候选节点的欧式距离构造 `distance_bias`
- 支持至少以下参数：
  - `--dar-enabled`
  - `--dar-k`
  - `--dar-alpha`
  - `--dar-log-nearest 1/0`

建议实现：

- 先不改原 `LCModel` 文件。
- 在新评测脚本中复制一份轻量推理路径，或写 wrapper 包住 decoder forward。
- 第一版只做 **eval-time DAR**，不做训练期注入。

Milestone：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/evaluate_lc_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --test-data Project/data/val/tsp50_ood_val_16.txt   --node-cnt 50   --pomo-size 50   --dar-enabled 1   --dar-k 10   --dar-alpha 1.0   --device cuda:0
```

可验证成果：

- 评测脚本能在 `DAR off` / `DAR on` 两种模式都正常跑完。
- 输出 JSON 中包含 `dar_k`、`dar_alpha` 和最终 `avg_cost/gap/time`。
- 至少在一个验证集上出现可观测差异，不论更好还是更差。

本次新增内容：

- `Project/experiments/lc_dar_elg/model/dar_wrapper.py`
- `Project/experiments/lc_dar_elg/evaluate_lc_dar.py`
- `Project/experiments/lc_dar_elg/eval_all_lc_dar.py`

本次实际执行命令：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --dar-enabled 0   --dar-k 10   --dar-alpha 1.0   --results-dir Project/experiments/lc_dar_elg/results/step2_dar_off_k10_a1

conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --dar-enabled 1   --dar-k 10   --dar-alpha 1.0   --results-dir Project/experiments/lc_dar_elg/results/step2_dar_on_k10_a1
```

本次实验结果（DAR off vs DAR on, `k=10`, `alpha=1.0`）：

| Dataset | DAR off cost | DAR on cost | Cost delta | DAR off gap | DAR on gap | Gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 12.8206 | 6.6961 | -6.1245 | 126.30% | 18.11% | -108.19% |
| tsp50_ood | 9.9655 | 5.7463 | -4.2191 | 105.72% | 19.04% | -86.68% |
| tsp100_uniform | 21.9777 | 9.1210 | -12.8567 | 181.34% | 16.64% | -164.70% |

可验证成果对应结果：

- **评测脚本能在 `DAR off` / `DAR on` 两种模式都正常跑完**：已完成。结果目录分别为：
  - `Project/experiments/lc_dar_elg/results/step2_dar_off_k10_a1/`
  - `Project/experiments/lc_dar_elg/results/step2_dar_on_k10_a1/`
- **输出 JSON 中包含 `dar_k`、`dar_alpha` 和最终 `avg_cost/gap/time`**：已完成。JSON 输出位于上述两个目录及 `Project/experiments/lc_dar_elg/results/step2_comparison.json`。
- **至少在一个验证集上出现可观测差异**：已完成，而且差异非常显著。当前 `k=10, alpha=1.0` 下，DAR 在三个验证集上都明显优于 DAR off。

当前结论：

- 对当前弱 baseline checkpoint 而言，DAR 是一个非常强的 inference-time patch。
- 它不仅改善 OOD 和 TSP100，也显著改善 TSP50 uniform。

长训练 baseline 复查（80 epoch checkpoint）：

为排除“DAR 只是修补弱 baseline”的可能，本次额外使用上一次长训练得到的 checkpoint：

- `Project/experiments/lc_leader/checkpoints/long_baseline_e80_b50_seed20260522_gpu5/best_model.pth`

本次实际执行命令：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py   --checkpoint Project/experiments/lc_leader/checkpoints/long_baseline_e80_b50_seed20260522_gpu5/best_model.pth   --device cuda:0   --dar-enabled 0   --dar-k 10   --dar-alpha 1.0   --results-dir Project/experiments/lc_dar_elg/results/step2_long_baseline_dar_off_k10_a1

conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py   --checkpoint Project/experiments/lc_leader/checkpoints/long_baseline_e80_b50_seed20260522_gpu5/best_model.pth   --device cuda:0   --dar-enabled 1   --dar-k 10   --dar-alpha 1.0   --results-dir Project/experiments/lc_dar_elg/results/step2_long_baseline_dar_on_k10_a1
```

长训练 baseline 上的实验结果（DAR off vs DAR on, `k=10`, `alpha=1.0`）：

| Dataset | DAR off cost | DAR on cost | Cost delta | DAR off gap | DAR on gap | Gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 5.9426 | 5.9025 | -0.0401 | 4.80% | 4.09% | -0.71% |
| tsp50_ood | 5.2612 | 5.1400 | -0.1212 | 8.95% | 6.34% | -2.61% |
| tsp100_uniform | 8.6457 | 8.4207 | -0.2250 | 10.55% | 7.69% | -2.86% |

复查结论：

- DAR 的收益并不只来自“基线太弱”。
- 在长训练 baseline 上，DAR 依然稳定改善三验证集，尤其对 `tsp50_ood` 和 `tsp100_uniform` 仍有较明确收益。
- 但长训练 baseline 上的改善幅度已经明显小于弱 baseline，这说明先前的超大提升中确实有一部分来自“短训 baseline 本身过差”。
- 因此后续 Step 3 的超参扫描，应优先以长训练 baseline 为主对象，避免被弱 baseline 的放大效应误导。

- 下一步需要做 Step 3 超参扫描，确认这种提升是否对 `K` 和 `alpha` 稳定，而不是只在单点参数上偶然成立。

## Step 3：做 DAR 小规模超参扫描

目标：

- 找到适合当前 LC baseline 的 DAR 超参数，而不是直接照搬论文。

需要完成：

- 至少扫描：
  - `K in {5, 10, 20, 50}`
  - `alpha in {0.25, 0.5, 1.0, 2.0}`
- 三个验证集都评估
- 生成一张汇总表

Milestone：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py \
  --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth \
  --device cuda:0 \
  --sweep-k 5,10,20,50 \
  --sweep-alpha 0.25,0.5,1.0,2.0
```

可验证成果：

- 输出 `summary.json` 和 `comparison.md`
- 能回答三个问题：
  - DAR 是否稳定提升 OOD？
  - DAR 是否稳定提升 TSP100？
  - 最优 K/alpha 是否和训练规模 50 节点有关？

## Step 4：实现 ELG-lite 的局部策略

目标：

- 在当前 LC/POMO 架构上加入一个 TSP-only 的 local policy。

需要完成：

- 新建 `Project/experiments/lc_dar_elg/model/local_policy.py`
- 每一步从当前节点提取 K 个最近未访问邻居
- 为每个邻居构造局部特征，建议第一版使用：
  - 相对坐标 `(dx, dy)`
  - 相对距离 `r`
  - 距离排序 index 的 positional encoding
- 局部策略输出每个候选节点的 `u_local`
- 对非局部邻居位置填 `0` 或 `-inf`，按实现方案固定

建议第一版简化：

- 不完整复现 ELG 的所有细节，先做一个 2~3 层 MLP / 小 Transformer 的 local scorer
- 先只支持 TSP，不引入 CVRP 容量信息

Milestone：

```bash
conda run -n amla_tsp python -m py_compile \
  Project/experiments/lc_dar_elg/model/local_policy.py
```

可验证成果：

- 局部策略模块可以独立 import
- 给定 `(coords, current_node, visited_mask)` 能输出 `(batch, pomo, node)` 的局部 score
- 对未入选的非邻居节点，mask 逻辑清晰且无 shape 错误

## Step 5：实现 ELG-lite 训练入口

目标：

- 新建联合训练脚本，把全局 score 和局部 score 融合起来训练。

需要完成：

- 新建 `Project/experiments/lc_dar_elg/train_lc_elg.py`
- 复用 `train_lc_leader.py` 的 CLI、checkpoint、seed、validation 逻辑
- 支持参数：
  - `--local-k`
  - `--local-policy-dim`
  - `--global-distance-penalty`
  - `--joint-train`
  - `--pretrain-global-epochs`
- 训练时用：

```text
u_global_tilde = u_global + distance_penalty
u_ens = u_global_tilde + beta * u_local
loss = POMO-style REINFORCE on ensemble policy
```

Milestone：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n amla_tsp \
  python Project/experiments/lc_dar_elg/train_lc_elg.py \
  --run-name elg_smoke \
  --epochs 1 \
  --batches-per-epoch 2 \
  --batch-size 4 \
  --local-k 10 \
  --device cuda:0
```

可验证成果：

- 1 epoch smoke run 成功
- 能保存 `best_model.pth`
- 日志能打印：
  - `global_score` 或其统计量
  - `local_score` 或其统计量
  - `ensemble loss`
  - `val cost`

## Step 6：做 ELG-lite 与 baseline 的同预算对比

目标：

- 判断局部策略是否比单纯 baseline 更稳地改善 OOD / TSP100。

建议预算：

```text
epochs = 20
batches_per_epoch = 50
batch_size = 64
val_interval = 5
seed = 固定
```

Milestone：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n amla_tsp \
  python Project/experiments/lc_dar_elg/train_lc_elg.py \
  --run-name elg_e20_b50 \
  --epochs 20 \
  --batches-per-epoch 50 \
  --batch-size 64 \
  --val-interval 5 \
  --local-k 10 \
  --device cuda:0
```

可验证成果：

- 训练曲线和 checkpoint 正常保存
- 三验证集评测结果写入 `results/elg_e20_b50/`
- 形成 baseline vs ELG-lite 表格

## Step 7：组合 ELG-lite + DAR

目标：

- 在 ELG-lite 训练出的 checkpoint 上，再叠加 DAR 推理修正。

需要完成：

- `evaluate_lc_dar.py` 支持加载 ELG-lite checkpoint
- 比较四组：
  - baseline
  - baseline + DAR
  - ELG-lite
  - ELG-lite + DAR

Milestone：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py \
  --checkpoint Project/experiments/lc_dar_elg/checkpoints/elg_e20_b50/best_model.pth \
  --device cuda:0 \
  --dar-enabled 1 \
  --dar-k 10 \
  --dar-alpha 1.0
```

可验证成果：

- 生成四组对比表
- 能回答：
  - DAR 单独是否有效？
  - ELG-lite 单独是否有效？
  - 组合后是否叠加收益，还是互相抵消？

## Step 8：完成最小消融

目标：

- 确认真正带来收益的是哪一部分。

最低要求的消融：

| 实验 | Local Policy | Global Distance Penalty | DAR |
| --- | --- | --- | --- |
| A | 关 | 关 | 关 |
| B | 关 | 关 | 开 |
| C | 开 | 开 | 关 |
| D | 开 | 开 | 开 |

Milestone：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py --preset ablation
```

可验证成果：

- `ablation_summary.json`
- `ablation.md`
- 至少能判断收益主要来自：
  - 推理期几何 bias
  - 训练期局部策略
  - 或两者组合

## Step 9：决定是否值得作为主线 close 或继续

建议 close / continue 判据：

- 如果 `baseline + DAR` 已经能稳定改善 `tsp50_ood` 和 `tsp100_uniform`，这条路线值得继续。
- 如果 `ELG-lite` 在同预算下仍然几乎没有提升，则不建议继续深挖完整 ELG 复现。
- 如果 `ELG-lite + DAR` 的提升显著高于单独 DAR，说明“训练期局部策略 + 推理期几何重塑”的复合点成立，可以作为最终项目主线。

最终可交付的最低版本：

1. 复用 baseline 三验证集结果。
2. 实现并验证 DAR inference patch。
3. 做 DAR 超参扫描。
4. 实现 ELG-lite smoke run。
5. 给出 baseline / DAR / ELG-lite / ELG-lite+DAR 的对比表。

## 推荐最终表格

| 方法 | TSP-50 Uniform Gap | TSP-50 OOD Gap | TSP-100 Gap | Time |
| --- | --- | --- | --- | --- |
| LC baseline | 待填 | 待填 | 待填 | 待填 |
| LC + DAR | 待填 | 待填 | 待填 | 待填 |
| LC + ELG-lite | 待填 | 待填 | 待填 | 待填 |
| LC + ELG-lite + DAR | 待填 | 待填 | 待填 | 待填 |

## 当前结论

和上一次相比，这条路线更值得做：

- DAR 是低风险、强可验证、直接作用于 attention/logit 的方法。
- ELG 是更重的训练增强，但与 POMO 风格骨架兼容。
- 两者结合形成了一个合理、可解释、且有官方代码可对照的工作点。

如果后续时间有限，优先级建议是：

1. 先做 DAR。
2. 再做 ELG-lite。
3. 最后再做组合与消融。
