# Step 7 Four-Way Comparison

All results use the same epoch-20 training budget. DAR uses `k=10, alpha=4.0, dar_log_nearest=1`.

| Method | TSP50 Uniform Gap | TSP50 OOD Gap | TSP100 Gap | Total time |
| --- | ---: | ---: | ---: | ---: |
| LC baseline | 8.23% | 11.93% | 16.38% | 4.07s |
| LC + DAR | 5.53% | 7.93% | 7.57% | 8.02s |
| LC + ELG-lite | 3.10% | 3.70% | 6.61% | 17.77s |
| LC + ELG-lite + DAR | 3.27% | 3.78% | 6.97% | 20.11s |

| Method | TSP50 Uniform delta | TSP50 OOD delta | TSP100 delta |
| --- | ---: | ---: | ---: |
| LC baseline | 0.00% | 0.00% | 0.00% |
| LC + DAR | -2.70% | -3.99% | -8.81% |
| LC + ELG-lite | -5.13% | -8.23% | -9.77% |
| LC + ELG-lite + DAR | -4.96% | -8.15% | -9.41% |
