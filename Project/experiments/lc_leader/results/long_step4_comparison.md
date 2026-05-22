# Long Step 4 Leader Reward Ablation

Both runs used CUDA_VISIBLE_DEVICES=5, 80 epochs, 50 batches/epoch, batch size 64, seed 20260522, and validation every 5 epochs.

## Final Three-Set Evaluation

| Dataset | Baseline cost | Leader cost | Cost delta | Baseline gap | Leader gap | Gap delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tsp50_uniform | 5.9426 | 5.9297 | -0.0129 | 4.80% | 4.58% | -0.22% |
| tsp50_ood | 5.2612 | 5.1812 | -0.0800 | 8.95% | 7.23% | -1.72% |
| tsp100_uniform | 8.6457 | 8.5995 | -0.0462 | 10.55% | 9.97% | -0.58% |

## Validation Cost Curve

| Epoch | Baseline val cost | Leader val cost |
| ---: | ---: | ---: |
| 5 | 6.3993 | 6.3776 |
| 10 | 6.2718 | 6.2583 |
| 15 | 6.1950 | 6.1540 |
| 20 | 6.1356 | 6.1011 |
| 25 | 6.0872 | 6.0632 |
| 30 | 6.0661 | 6.0450 |
| 35 | 6.0377 | 6.0171 |
| 40 | 6.0118 | 6.0094 |
| 45 | 5.9796 | 5.9806 |
| 50 | 5.9746 | 5.9792 |
| 55 | 5.9689 | 5.9687 |
| 60 | 5.9551 | 5.9619 |
| 65 | 5.9476 | 5.9537 |
| 70 | 5.9473 | 5.9333 |
| 75 | 5.9528 | 5.9297 |
| 80 | 5.9426 | 5.9321 |

The validation curve is still improving slowly at 80 epochs, but the last 10 epochs are near a plateau. Leader Reward keeps a small but consistent advantage in this run.
