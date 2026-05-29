# LC Final Submission Package

这个目录是按照 `Project/INSTRUCTION.md` 中 **LC 范式提交目录** 整理后的最终版本。只保留作业要求的那套结构：

```text
baselines/lc_baseline/
├── model/
│   ├── __init__.py
│   ├── lc_model.py
│   └── tsp_env.py
├── checkpoints/
│   └── best_model.pth
├── evaluate_lc.py
├── train_lc.py
└── README.md
```

最终方法是 **LC baseline + ELG-lite local policy + tuned DAR inference**：

- `local_k=10`
- `local_score_weight=1.0`
- `global_distance_penalty=0.5`
- `dar_enabled=1`
- `dar_k=20`
- `dar_alpha=0.5`
- `dar_log_nearest=1`

## 1. 环境配置

推荐新建独立环境：

```bash
conda create -n amla_tsp python=3.8 -y
conda activate amla_tsp
pip install -r ../../requirements.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
```

## 2. 自测评估

在当前目录下运行：

```bash
conda run -n amla_tsp python evaluate_lc.py
```

评估脚本满足作业要求：

- 定义了全局 `model_params`
- 定义了全局 `env_params`
- `from model import LCModel, TSPEnv` 可直接导入
- 自动加载 `checkpoints/best_model.pth`

如果只想评测 ELG-lite 而不叠加 DAR，把 `evaluate_lc.py` 中的 `DAR_ENABLED` 改成 `0` 即可。

## 3. 训练复现

训练脚本不再依赖 `experiments/` 目录，直接在当前目录可运行：

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n amla_tsp python train_lc.py   --epochs 20   --batches-per-epoch 50   --batch-size 64   --val-interval 5   --seed 20260522   --local-k 10   --local-policy-dim 128   --local-score-weight 1.0   --global-distance-penalty 0.5   --distance-k 10   --device cuda:0
```

说明：

- 训练阶段默认 `dar_enabled=0`，对应之前的实验设定：先训练 ELG-lite，再在推理时叠加 tuned DAR。
- 训练完成后会将最新最好权重写到 `checkpoints/best_model.pth`。

## 4. 与历史实验目录的关系

`Project/experiments/` 和 `Project/final_lc_elg_dar/` 只保留为历史记录。当前这个 `lc_baseline/` 目录已经是自洽提交包；即使不依赖那两个目录，评测接口和训练脚本也能正常运行。
