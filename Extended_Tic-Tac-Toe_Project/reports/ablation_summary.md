# Ablation & Research Question Empirical Summary (Task T4.4)

## ABLATION_1 (`mode=full`, `matches=316`, `log=ablation_ab_pruning.jsonl`)

### Controlled State Probes (`4x4, K=4`)
| Depth | MM Raw Avg Nodes | AB (No Order) Avg Nodes | AB + Order Avg Nodes | AB Reduction (%) | AB + Order Reduction (%) | Exact Root Value |
|:---:|---:|---:|---:|---:|---:|:---:|
| d=2 | 175.0 | 60.0 | 54.2 | 65.71% | 69.00% | YES |
| d=3 | 2071.0 | 353.0 | 322.0 | 82.96% | 84.45% | YES |
| d=4 | 23171.0 | 2035.5 | 1225.0 | 91.22% | 94.71% | YES |

### Paired Gameplay Comparison
| Variant | Games | Score [95% CI] | Mean Latency (ms) | P95 Latency (ms) | Avg Nodes |
|:---|---:|:---:|---:|---:|---:|
| `MM_Raw_d3` | 60 | 0.733 [0.675, 0.800] | 10.65 | 32.34 | 1161.3 |
| `AB_NoOrder_d3` | 60 | 0.717 [0.658, 0.783] | 2.35 | 6.53 | 255.1 |
| `AB_Order_d3` | 60 | 0.733 [0.675, 0.800] | 2.46 | 5.32 | 254.0 |
| `MM_Raw_d4` | 16 | 0.656 [0.562, 0.781] | 105.76 | 428.75 | 11368.8 |
| `AB_NoOrder_d4` | 60 | 0.708 [0.650, 0.775] | 10.95 | 30.24 | 1187.2 |
| `AB_Order_d4` | 60 | 0.667 [0.608, 0.725] | 7.88 | 19.32 | 805.6 |

## ABLATION_2 (`mode=full`, `matches=2760`, `log=ablation_rq1_hybrid_mcts.jsonl`)

| Candidate | Budget | Games vs Pool | Score [95% CI] | Mean (ms) | P95 (ms) | Avg Sims | Cache Hit |
|:---|:---:|---:|:---:|---:|---:|---:|---:|
| `Policy_Fast` | `none=0` | 120 | 0.542 [0.496, 0.588] | 0.35 | 0.60 | 0.0 | - |
| `Policy_Strong` | `none=0` | 120 | 0.525 [0.479, 0.575] | 0.34 | 0.45 | 0.0 | - |
| `Value_Greedy` | `none=0` | 120 | 0.296 [0.225, 0.371] | 0.43 | 0.57 | 0.0 | - |
| `PureMCTS_sim50` | `simulations=50` | 120 | 0.529 [0.471, 0.588] | 3.79 | 6.63 | 50.0 | - |
| `HybridRollout_sim50` | `simulations=50` | 120 | 0.583 [0.542, 0.625] | 8.78 | 20.05 | 50.0 | 47.0% |
| `HybridMixed_sim50` | `simulations=50` | 120 | 0.575 [0.533, 0.621] | 8.99 | 23.55 | 50.0 | 60.7% |
| `PureMCTS_sim150` | `simulations=150` | 120 | 0.604 [0.571, 0.642] | 10.75 | 19.94 | 150.0 | - |
| `HybridRollout_sim150` | `simulations=150` | 120 | 0.579 [0.537, 0.621] | 22.81 | 53.46 | 150.0 | 47.3% |
| `HybridMixed_sim150` | `simulations=150` | 120 | 0.567 [0.521, 0.613] | 23.67 | 62.51 | 150.0 | 63.9% |
| `PureMCTS_sim400` | `simulations=400` | 120 | 0.621 [0.583, 0.658] | 28.51 | 53.13 | 400.0 | - |
| `HybridRollout_sim400` | `simulations=400` | 120 | 0.558 [0.521, 0.596] | 55.24 | 134.70 | 400.0 | 47.1% |
| `HybridMixed_sim400` | `simulations=400` | 120 | 0.600 [0.562, 0.637] | 56.74 | 155.85 | 400.0 | 62.8% |
| `PureMCTS_time10ms` | `time_ms=10.0` | 120 | 0.613 [0.571, 0.654] | 9.27 | 10.16 | 190.7 | - |
| `HybridRollout_time10ms` | `time_ms=10.0` | 120 | 0.575 [0.533, 0.617] | 9.41 | 10.40 | 144.0 | 51.2% |
| `HybridMixed_time10ms` | `time_ms=10.0` | 120 | 0.583 [0.542, 0.625] | 9.47 | 10.48 | 145.9 | 68.7% |
| `PureMCTS_time50ms` | `time_ms=50.0` | 120 | 0.608 [0.571, 0.646] | 28.49 | 50.21 | 396.8 | - |
| `HybridRollout_time50ms` | `time_ms=50.0` | 120 | 0.583 [0.542, 0.625] | 33.38 | 50.47 | 313.2 | 52.1% |
| `HybridMixed_time50ms` | `time_ms=50.0` | 120 | 0.579 [0.533, 0.625] | 34.04 | 50.52 | 312.1 | 69.0% |

## ABLATION_3 (`mode=full`, `matches=2080`, `log=ablation_rq2_seat_advantage.jsonl`)

| Board Variant | Matches | X Win% | O Win% | Draw% | Score(X) | Score(O) | ΔScore (X-O) [95% CI] |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| `3x3_k3` | 840 | 23.8% | 12.6% | 63.6% | 0.556 | 0.444 | +0.112 [+0.073, +0.152] |
| `4x4_k3` | 400 | 76.5% | 23.5% | 0.0% | 0.765 | 0.235 | +0.530 [+0.445, +0.610] |
| `4x4_k4` | 840 | 14.2% | 11.8% | 74.1% | 0.512 | 0.488 | +0.024 [-0.011, +0.058] |

## ABLATION_4 (`mode=full`, `matches=3600`, `log=ablation_rq3_rl_generalization.jsonl`)

### Board `4x4_k4`
| Candidate | Group | Overall Score [95% CI] | Unseen MCTS Score | Unseen All Score | Baseline Score | Mean (ms) |
|:---|:---:|:---:|---:|---:|---:|---:|
| `SL_FastPolicy` | Supervised | 0.514 [0.494, 0.533] | 0.425 | 0.463 | 0.637 | 0.32 |
| `SL_StrongPolicy` | Supervised | 0.497 [0.476, 0.518] | 0.344 | 0.422 | 0.644 | 0.35 |
| `SL_ValueGreedy` | Supervised | 0.307 [0.272, 0.343] | 0.256 | 0.253 | 0.500 | 0.45 |
| `RL_DQN_seed42` | RL | 0.582 [0.562, 0.601] | 0.537 | 0.519 | 0.706 | 0.26 |
| `RL_DQN_seed123` | RL | 0.621 [0.599, 0.644] | 0.575 | 0.537 | 0.719 | 0.23 |
| `RL_DQN_seed999` | RL | 0.556 [0.539, 0.574] | 0.550 | 0.525 | 0.700 | 0.23 |

- **RL 3-Seed Stability (`[42, 123, 999]`)**: Overall `0.586 ± 0.027`, Unseen MCTS `0.554 ± 0.016`

### Board `3x3_k3`
| Candidate | Group | Overall Score [95% CI] | Unseen MCTS Score | Unseen All Score | Baseline Score | Mean (ms) |
|:---|:---:|:---:|---:|---:|---:|---:|
| `SL_FastPolicy` | Supervised | 0.476 [0.444, 0.507] | 0.556 | 0.528 | 0.713 | 0.24 |
| `SL_StrongPolicy` | Supervised | 0.485 [0.454, 0.515] | 0.581 | 0.541 | 0.725 | 0.25 |
| `SL_ValueGreedy` | Supervised | 0.469 [0.438, 0.501] | 0.506 | 0.503 | 0.731 | 0.29 |
| `RL_TabQ_seed42` | RL | 0.653 [0.629, 0.676] | 0.594 | 0.547 | 0.719 | 0.03 |
| `RL_TabQ_seed123` | RL | 0.661 [0.637, 0.686] | 0.613 | 0.556 | 0.738 | 0.03 |
| `RL_TabQ_seed999` | RL | 0.664 [0.640, 0.689] | 0.650 | 0.575 | 0.713 | 0.03 |

- **RL 3-Seed Stability (`[42, 123, 999]`)**: Overall `0.659 ± 0.005`, Unseen MCTS `0.619 ± 0.023`

