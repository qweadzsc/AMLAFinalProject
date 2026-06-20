# LC 提交包

## 环境配置

```bash
conda create -n amla_tsp python=3.8 -y
conda activate amla_tsp
pip install -r requirements.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
```

## 1. 方法概述

本项目选择的是 **LC（Local Construction）范式**，即基于自回归强化学习的 TSP 构造式求解方法。基础骨架沿用课程提供的 LC baseline：使用 Transformer 编码器提取节点表示，再用 attention decoder 逐步选择下一访问节点，并用 POMO 风格的多起点 rollout 进行训练。

在 baseline 的基础上，最终版本加入了两项核心改进：

1. **ELG-lite 局部策略增强**：在全局 decoder score 之外，引入一个轻量的局部策略打分器，只关注当前节点附近的若干近邻，用于补充更稳定的局部几何归纳偏置。
2. **DAR 推理期距离重塑**：在推理阶段对 logits 叠加基于距离的 bias，使模型在候选节点较多、泛化更困难时，优先考虑更合理的近邻移动方向。

最终提交版采用的是 **LC baseline + ELG-lite local policy + tuned DAR inference** 的组合方案。

## 2. 模型架构

整体结构如下：

- **编码器**：3 层 Transformer encoder，对二维坐标进行嵌入并输出节点表示。
- **解码器**：保留 baseline 的 multi-head attention + single-head score 结构，输出每一步选择下一节点的 logits。
- **局部策略模块（ELG-lite）**：
  - 输入当前节点与所有候选节点的相对几何关系；
  - 仅对最近的 `K` 个邻居进行局部打分；
  - 输出局部 score，并与全局 score 相加。
- **距离偏置模块**：
  - `global_distance_penalty` 在训练和评测中都可启用；
  - `DAR` 在最终评测配置中额外开启，用于进一步 reshape logits。

与 baseline 相比，主要差异有三点：

1. baseline 只有全局 policy，本实现增加了显式的 **local policy scorer**；
2. baseline 的 decoder logits 只来自网络本身，本实现加入了 **distance-aware bias**；
3. 最终版本将“训练期局部增强”和“推理期距离重塑”组合起来，而不是只做单一修改。

## 3. 训练配置

最终训练脚本位于 `baselines/lc_baseline/train_lc.py`，关键默认超参数如下：

- `embedding_dim = 128`
- `num_att_layers = 3`
- `num_heads = 8`
- `qkv_dim = 16`
- `ff_hidden_dim = 512`
- `logit_clipping = 10`
- `node_cnt = 50`
- `pomo_size = 50`
- `local_policy_dim = 128`
- `local_k = 10`
- `local_score_weight = 1.0`
- `global_distance_penalty = 0.5`
- `distance_k = 10`
- `learning_rate = 1e-4`
- `batch_size = 64`
- `epochs = 20`
- `batches_per_epoch = 50`
- `val_interval = 5`
- `seed = 20260522`

训练阶段默认关闭 `dar_enabled`，即先训练 ELG-lite 局部增强模型；DAR 作为推理期增强，在评测脚本中默认开启。

一个典型训练命令如下：

```bash
conda run -n amla_tsp python baselines/lc_baseline/train_lc.py   --epochs 20   --batches-per-epoch 50   --batch-size 64   --val-interval 5   --seed 20260522   --local-k 10   --local-policy-dim 128   --local-score-weight 1.0   --global-distance-penalty 0.5   --distance-k 10   --device cuda:0
```

## 4. 实验结果

### 4.1 验证集主结果

当前提交版 `evaluate_lc.py` 在 `tsp50_uniform_val_128` 上的结果如下：

| 方法 | Average Cost | Average Optimal | Optimality Gap | Avg Time / Instance |
| --- | ---: | ---: | ---: | ---: |
| 初始 LC baseline（早期固定评测） | 12.8206 | 5.6709 | 126.30% | - |
| 最终提交版（ELG-lite + DAR） | 5.8414 | 5.6709 | 3.01% | 0.0999s |

从最终提交版相对于初始 baseline 的结果看，求解质量有明显提升，且推理耗时仍保持在可接受范围内。

### 4.2 中间对比实验

完整实验过程中还做过若干组对比，结论如下：

- **DAR 单独加入 baseline**：在 TSP50 uniform / OOD 上有帮助，但对更大规模的收益有限。
- **ELG-lite 单独使用**：比同预算 baseline 更稳定，说明局部策略增强本身有效。
- **ELG-lite + DAR**：在 TSP50 上可进一步小幅改善，但增益不算大，更像稳健的小修正而不是决定性提升。

## 5. 设计动机与分析

首先观察到的问题是：

1. **训练长度对结果非常敏感**。短训练时 baseline gap 极大，说明必须先区分“训练不充分”和“方法本身不行”。
2. **即使延长训练，泛化问题依然存在**。尤其在 OOD 分布和跨规模设置下，baseline 的表现仍然不够稳定。

基于这些现象，设计思路逐步收敛为：

- 只改 reward shaping 不足以解决问题；
- 只在推理期做距离 bias 有帮助，但不足以完全解决泛化问题；
- 需要给模型本身加入更明确的局部几何归纳偏置。

因此最终落在“**局部策略增强 + 距离重塑**”这条路线：

- ELG-lite 对应“训练期学到更可迁移的局部模式”；
- DAR 对应“推理期抑制候选注意力分散”。

实验现象与理论预期是基本一致的：

- 局部策略确实能带来更稳的改进；
- DAR 作为 inference-time patch 能在部分验证集上继续改善，但提升幅度有限；

## 6. 改进方向

后续如果继续做，可以考虑以下几个方向：

1. **更系统的超参数扫描**：例如局部近邻数、局部策略维度、distance penalty 权重等，还可以做更细粒度搜索。
2. **训练与推理的一致化**：当前 DAR 主要在推理阶段使用，后续可以尝试将距离偏置更自然地并入训练过程。
3. **更强的局部策略建模**：目前的 ELG-lite 仍是轻量实现，可以进一步尝试更完整的位置编码和局部上下文建模。
4. **跨规模联合训练**：当前训练仍然以 TSP50 为主，后续可以直接尝试混合规模训练，进一步提升 TSP100 等场景的泛化能力。
5. **更完整的消融**：将 local policy、global distance penalty、DAR 分别关掉，做更系统的 ablation table，有助于更清晰地量化每一部分的真实贡献。
