# LC Submission Package

This folder is the final self-contained LC submission package. It matches the directory layout required by `Project/INSTRUCTION.md` and can be tested directly from this folder.

## Directory

```text
lc_baseline/
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

## Method

The final model keeps the LC baseline interface and adds two changes inside `model/lc_model.py`:

- an ELG-lite local policy scorer
- a tuned DAR inference bias

Default evaluation configuration:

- `local_k=10`
- `local_score_weight=1.0`
- `global_distance_penalty=0.5`
- `dar_enabled=1`
- `dar_k=20`
- `dar_alpha=0.5`
- `dar_log_nearest=1`

## Environment

```bash
conda create -n amla_tsp python=3.8 -y
conda activate amla_tsp
pip install -r ../../requirements.txt
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
```

## Evaluation

Run from this folder:

```bash
conda run -n amla_tsp python evaluate_lc.py
```

The script:

- defines global `model_params`
- defines global `env_params`
- loads `checkpoints/best_model.pth`
- supports direct `from model import LCModel, TSPEnv`

## Training

Run from this folder:

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n amla_tsp python train_lc.py \
  --epochs 20 \
  --batches-per-epoch 50 \
  --batch-size 64 \
  --val-interval 5 \
  --seed 20260522 \
  --local-k 10 \
  --local-policy-dim 128 \
  --local-score-weight 1.0 \
  --global-distance-penalty 0.5 \
  --distance-k 10 \
  --device cuda:0
```

Training uses the ELG-lite local policy path and keeps `dar_enabled=0` during training; DAR is applied in evaluation by default.
