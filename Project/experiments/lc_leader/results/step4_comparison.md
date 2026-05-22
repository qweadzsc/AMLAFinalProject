# Step 4 Leader Reward Ablation

Training budget for both runs: 5 epochs, 20 batches/epoch, batch size 64, seed 20260522.

| Dataset | Baseline cost | Leader cost | Cost delta | Baseline gap | Leader gap | Gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 6.6770 | 6.6380 | -0.0390 | 17.75% | 17.08% | -0.67% |
| tsp50_ood | 5.7407 | 5.7430 | +0.0023 | 18.84% | 18.91% | +0.08% |
| tsp100_uniform | 10.5370 | 10.3806 | -0.1565 | 34.77% | 32.75% | -2.02% |

Negative deltas indicate the Leader Reward run improved over the baseline run.
