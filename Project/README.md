# AMLA Final Project

本仓库最终选择的是 LC 路线，并将最终可提交版本整理在 `Project/final_lc_elg_dar/`。原始 `Project/baselines/` 和 `Project/data/` 未被修改；所有实验性实现与结果均放在 `Project/experiments/` 下。

## 1. 环境配置

推荐直接使用课程说明里的依赖，并统一放到 `amla_tsp` 环境中：

```bash
conda create -n amla_tsp python=3.8 -y
conda activate amla_tsp
pip install -r Project/requirements.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
```

## 2. 最终提交接口

最终接口目录：`Project/final_lc_elg_dar/`

其中：

- `model/lc_model.py`：保留 LC baseline 相同接口的最终模型。
- `model/tsp_env.py`：与 baseline 一致的环境接口。
- `evaluate_lc.py`：定义作业要求的 `model_params` 和 `env_params`，默认加载最终权重。
- `train_lc.py`：对训练脚本的包装入口，便于按作业目录结构定位。

默认最终版本为 `ELG-lite + tuned DAR`：

- `local_k=10`
- `local_score_weight=1.0`
- `global_distance_penalty=0.5`
- `dar_enabled=1`
- `dar_k=20`
- `dar_alpha=0.5`
- `dar_log_nearest=1`

运行最终接口自测：

```bash
cd Project/final_lc_elg_dar
conda run -n amla_tsp python evaluate_lc.py
```

如果只想评测 ELG-lite 而不叠加 DAR，把 `evaluate_lc.py` 里的 `model_params["dar_enabled"]` 改成 `0` 即可。

## 3. 主要实验目录

- Baseline 统一评测：`Project/experiments/lc_eval/`
- Leader Reward 尝试：`Project/experiments/lc_leader/`
- DAR / ELG / ELG+DAR 主线：`Project/experiments/lc_dar_elg/`

核心结果文件：

- Step 1 baseline：`Project/experiments/lc_eval/results/step1_baseline_refresh/summary.json`
- Step 3 DAR sweep：`Project/experiments/lc_dar_elg/results/step3_long_baseline_sweep/comparison.md`
- Step 6 ELG-lite：`Project/experiments/lc_dar_elg/results/step6_comparison.md`
- Step 7 ELG-lite + DAR sweep：`Project/experiments/lc_dar_elg/results/step7_elg_dar_sweep_e20_b50_seed20260522/comparison.md`

## 4. 全部实验复现命令

### Step 1：固定 baseline 评测

```bash
conda run -n amla_tsp python Project/experiments/lc_eval/eval_all_lc.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --results-dir Project/experiments/lc_eval/results/step1_baseline_refresh
```

### Step 2：DAR 独立推理验证

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --dar-enabled 1   --dar-k 10   --dar-alpha 1.0   --results-dir Project/experiments/lc_dar_elg/results/step2_dar_on_k10_a1
```

### Step 3：DAR 超参数扫描

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/sweep_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --results-dir Project/experiments/lc_dar_elg/results/step3_long_baseline_sweep
```

扩展 alpha 扫描：

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/sweep_dar.py   --checkpoint Project/baselines/lc_baseline/checkpoints/best_model.pth   --device cuda:0   --sweep-alpha 3.0,4.0,6.0,8.0   --results-dir Project/experiments/lc_dar_elg/results/step3_long_baseline_sweep_alpha_extended
```

### Step 4：ELG-lite 结构 smoke test

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n amla_tsp python Project/experiments/lc_dar_elg/train_lc_elg.py   --run-name elg_smoke_v2   --epochs 1   --batches-per-epoch 2   --batch-size 4   --val-interval 1   --local-k 10   --local-policy-dim 64   --local-score-weight 1.0   --global-distance-penalty 0.5   --joint-train 1   --pretrain-global-epochs 0   --device cuda:0
```

### Step 5：更长训练验证结构可跑通

```bash
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n amla_tsp python -u Project/experiments/lc_dar_elg/train_lc_elg.py   --run-name elg_e20_b50_seed20260522   --epochs 20   --batches-per-epoch 50   --batch-size 64   --val-interval 5   --seed 20260522   --local-k 10   --local-policy-dim 128   --local-score-weight 1.0   --global-distance-penalty 0.5   --joint-train 1   --pretrain-global-epochs 0   --device cuda:0
```

### Step 6：ELG-lite 同预算评测

```bash
conda run -n amla_tsp python Project/experiments/lc_dar_elg/eval_all_lc_elg.py   --checkpoint Project/experiments/lc_dar_elg/checkpoints/elg_e20_b50_seed20260522/best_model.pth   --device cuda:0   --results-dir Project/experiments/lc_dar_elg/results/step6_elg_e20_b50_seed20260522
```

### Step 7：ELG-lite + DAR 超参数扫描

```bash
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n amla_tsp python -u Project/experiments/lc_dar_elg/sweep_elg_dar.py   --checkpoint Project/experiments/lc_dar_elg/checkpoints/elg_e20_b50_seed20260522/best_model.pth   --device cuda:0   --sweep-k 5,10,20   --sweep-alpha 0.05,0.1,0.25,0.5,1.0,2.0,4.0   --results-dir Project/experiments/lc_dar_elg/results/step7_elg_dar_sweep_e20_b50_seed20260522
```
