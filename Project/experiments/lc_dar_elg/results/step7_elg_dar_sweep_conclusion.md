# ELG-lite DAR Hyperparameter Sweep Conclusion

Grid: `k=5,10,20`, `alpha=0.05,0.1,0.25,0.5,1.0,2.0,4.0`, `dar_log_nearest=1`.

| Setting | TSP50 Uniform Gap | TSP50 OOD Gap | TSP100 Gap | Mean Gap |
| --- | ---: | ---: | ---: | ---: |
| ELG-lite | 3.10% | 3.70% | 6.61% | 4.47% |
| ELG-lite + DAR best mean (`k=20,a=0.5`) | 3.01% | 3.51% | 6.67% | 4.40% |

| Dataset | Best run | Best gap | Delta vs ELG-lite |
| --- | --- | ---: | ---: |
| tsp50_uniform | k10_a0p5 | 3.01% | -0.09% |
| tsp50_ood | k20_a0p5 | 3.51% | -0.19% |
| tsp100_uniform | k5_a0p05 | 6.61% | 0.00% |

Conclusion: tuned DAR is not consistently harmful. Small alpha can slightly improve TSP50 uniform/OOD and the mean gap, but TSP100 is already saturated and tends to degrade once alpha grows.
