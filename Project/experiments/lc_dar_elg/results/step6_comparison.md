# Step 6 Same-Budget ELG-lite Comparison

Baseline: LC epoch-20 checkpoint from `long_baseline_e80_b50_seed20260522_gpu5`.
ELG-lite: `elg_e20_b50_seed20260522/best_model.pth`.

| Dataset | Baseline cost | ELG cost | Cost delta | Baseline gap | ELG gap | Gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 6.1356 | 5.8467 | -0.2889 | 8.23% | 3.10% | -5.13% |
| tsp50_ood | 5.4034 | 5.0105 | -0.3929 | 11.93% | 3.70% | -8.23% |
| tsp100_uniform | 9.1021 | 8.3356 | -0.7666 | 16.38% | 6.61% | -9.77% |
