# Consolidated Empirical Summary Tables (Extended Tic-Tac-Toe)

Generated directly from raw `logs/*.jsonl` files without manual edits.

## Figures Reference
- **Fig 1:** `figures/fig1_head_to_head_matrix.png` — Dual X/O Head-to-Head Matrix
- **Fig 2:** `figures/fig2_latency_and_nodes.png` — Log Latency (Mean & P95) & Search Cost
- **Fig 3:** `figures/fig3_ablation_rq1_and_alphabeta.png` — RQ1 Hybrid MCTS & Alpha-Beta Pruning
- **Fig 4:** `figures/fig4_ablation_rq2_seat_advantage.png` — RQ2 First-Mover Advantage
- **Fig 5:** `figures/fig5_ablation_rq3_learning_and_generalization.png` — RQ3 Learning Curves & Unseen Opponents

## 1. Official Tournament Summaries

| Log File | Board | Mode | Matches | Top Agent | Top Score [95% CI] | Seat ΔScore [95% CI] |
|:---|:---:|:---|---:|:---|:---:|:---:|
| `tournament_3x3_k3_default.jsonl` | `3x3 K=3` | `default` | 4200 | **L7_HybridMCTS** | 0.637 [0.624, 0.650] | +0.141 [+0.124, +0.159] |
| `tournament_4x4_k4_default.jsonl` | `4x4 K=4` | `default` | 4200 | **L4_MCTS** | 0.615 [0.603, 0.627] | +0.015 [-0.000, +0.031] |
| `tournament_4x4_k4_fixed_10ms.jsonl` | `4x4 K=4` | `fixed_10ms` | 4200 | **L4_MCTS** | 0.595 [0.581, 0.608] | +0.020 [+0.005, +0.038] |
| `tournament_4x4_k4_fixed_50ms.jsonl` | `4x4 K=4` | `fixed_50ms` | 4200 | **L4_MCTS** | 0.612 [0.600, 0.625] | +0.015 [-0.000, +0.031] |

### Leaderboard — `tournament_3x3_k3_default.jsonl` (`3x3 K=3`, `default`)

| Rank | Agent | W-D-L | Score [95% CI] | Score(X) | Score(O) | ΔScore | Mean (ms) | P95 (ms) | Avg Nodes | Avg Sims |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **L7_HybridMCTS** | 328-872-0 | 0.637 [0.624, 0.650] | 0.699 | 0.574 | +0.125 | 12.08 | 32.37 | 39.8 | 200.0 |
| 2 | **L6_TabularQ** | 318-882-0 | 0.632 [0.620, 0.645] | 0.700 | 0.565 | +0.135 | 0.04 | 0.08 | 1.0 | 0.0 |
| 3 | **L3_AlphaBeta** | 318-882-0 | 0.632 [0.620, 0.646] | 0.700 | 0.565 | +0.135 | 25.92 | 170.71 | 1406.3 | 0.0 |
| 4 | **L2_RuleBased** | 194-1006-0 | 0.581 [0.570, 0.590] | 0.593 | 0.569 | +0.023 | 0.10 | 0.14 | 15.6 | 0.0 |
| 5 | **L4_MCTS** | 200-824-176 | 0.510 [0.494, 0.526] | 0.593 | 0.427 | +0.165 | 15.81 | 36.34 | 103.1 | 200.0 |
| 6 | **L5_StrongPolicy** | 215-666-319 | 0.457 [0.438, 0.476] | 0.615 | 0.298 | +0.317 | 0.51 | 0.84 | 1.0 | 0.0 |
| 7 | **L1_Random** | 8-106-1086 | 0.051 [0.042, 0.060] | 0.095 | 0.007 | +0.088 | 0.04 | 0.07 | 1.0 | 0.0 |

### Leaderboard — `tournament_4x4_k4_default.jsonl` (`4x4 K=4`, `default`)

| Rank | Agent | W-D-L | Score [95% CI] | Score(X) | Score(O) | ΔScore | Mean (ms) | P95 (ms) | Avg Nodes | Avg Sims |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **L4_MCTS** | 282-911-7 | 0.615 [0.603, 0.627] | 0.616 | 0.613 | +0.003 | 21.54 | 61.83 | 156.7 | 200.0 |
| 2 | **L2_RuleBased** | 203-997-0 | 0.585 [0.574, 0.595] | 0.593 | 0.577 | +0.016 | 0.13 | 0.22 | 26.4 | 0.0 |
| 3 | **L6_DQN** | 179-995-26 | 0.564 [0.552, 0.575] | 0.553 | 0.574 | -0.021 | 0.40 | 0.85 | 1.0 | 0.0 |
| 4 | **L3_AlphaBeta** | 148-1052-0 | 0.562 [0.553, 0.571] | 0.572 | 0.552 | +0.020 | 12.75 | 46.48 | 740.6 | 0.0 |
| 5 | **L7_HybridMCTS** | 181-980-39 | 0.559 [0.548, 0.571] | 0.557 | 0.562 | -0.005 | 47.79 | 214.21 | 87.0 | 200.0 |
| 6 | **L5_StrongPolicy** | 131-983-86 | 0.519 [0.507, 0.530] | 0.532 | 0.505 | +0.027 | 0.73 | 1.72 | 1.0 | 0.0 |
| 7 | **L1_Random** | 8-218-974 | 0.098 [0.086, 0.110] | 0.129 | 0.066 | +0.063 | 0.06 | 0.08 | 1.0 | 0.0 |

### Leaderboard — `tournament_4x4_k4_fixed_10ms.jsonl` (`4x4 K=4`, `fixed_10ms`)

| Rank | Agent | W-D-L | Score [95% CI] | Score(X) | Score(O) | ΔScore | Mean (ms) | P95 (ms) | Avg Nodes | Avg Sims |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **L4_MCTS** | 275-877-48 | 0.595 [0.581, 0.608] | 0.609 | 0.580 | +0.029 | 8.77 | 10.32 | 83.1 | 124.4 |
| 2 | **L3_AlphaBeta** | 213-987-0 | 0.589 [0.578, 0.600] | 0.595 | 0.583 | +0.012 | 7.80 | 14.41 | 455.6 | 0.0 |
| 3 | **L2_RuleBased** | 210-990-0 | 0.588 [0.576, 0.598] | 0.600 | 0.575 | +0.025 | 0.07 | 0.11 | 26.4 | 0.0 |
| 4 | **L6_DQN** | 193-975-32 | 0.567 [0.555, 0.579] | 0.561 | 0.573 | -0.013 | 0.40 | 1.05 | 1.0 | 0.0 |
| 5 | **L7_HybridMCTS** | 182-974-44 | 0.557 [0.546, 0.570] | 0.557 | 0.557 | +0.000 | 8.81 | 10.58 | 28.5 | 105.8 |
| 6 | **L5_StrongPolicy** | 132-980-88 | 0.518 [0.507, 0.530] | 0.532 | 0.504 | +0.028 | 0.65 | 1.77 | 1.0 | 0.0 |
| 7 | **L1_Random** | 7-193-1000 | 0.086 [0.075, 0.098] | 0.117 | 0.056 | +0.061 | 0.03 | 0.06 | 1.0 | 0.0 |

### Leaderboard — `tournament_4x4_k4_fixed_50ms.jsonl` (`4x4 K=4`, `fixed_50ms`)

| Rank | Agent | W-D-L | Score [95% CI] | Score(X) | Score(O) | ΔScore | Mean (ms) | P95 (ms) | Avg Nodes | Avg Sims |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **L4_MCTS** | 273-922-5 | 0.612 [0.600, 0.625] | 0.613 | 0.611 | +0.002 | 30.22 | 50.42 | 146.3 | 189.3 |
| 2 | **L2_RuleBased** | 204-996-0 | 0.585 [0.574, 0.595] | 0.593 | 0.577 | +0.017 | 0.11 | 0.20 | 26.4 | 0.0 |
| 3 | **L6_DQN** | 181-988-31 | 0.562 [0.551, 0.574] | 0.550 | 0.575 | -0.025 | 0.76 | 1.21 | 1.0 | 0.0 |
| 4 | **L3_AlphaBeta** | 147-1053-0 | 0.561 [0.552, 0.571] | 0.571 | 0.552 | +0.019 | 12.90 | 47.05 | 937.5 | 0.0 |
| 5 | **L7_HybridMCTS** | 179-987-34 | 0.560 [0.549, 0.572] | 0.559 | 0.562 | -0.002 | 32.88 | 51.65 | 53.2 | 153.0 |
| 6 | **L5_StrongPolicy** | 131-984-85 | 0.519 [0.508, 0.531] | 0.531 | 0.507 | +0.023 | 1.27 | 1.94 | 1.0 | 0.0 |
| 7 | **L1_Random** | 8-224-968 | 0.100 [0.089, 0.112] | 0.136 | 0.064 | +0.072 | 0.05 | 0.08 | 1.0 | 0.0 |

## 2. Ablation Studies Summary (Alpha-Beta, RQ1, RQ2, RQ3)

### ABLATION_1 (`ablation_ab_pruning.jsonl`, matches=316)

| Depth | MM Nodes | AB No-Order Nodes | AB + Order Nodes | AB+Order Reduction |
|:---:|---:|---:|---:|---:|
| d=2 | 175.0 | 60.0 | 54.2 | -69.00% |
| d=3 | 2071.0 | 353.0 | 322.0 | -84.45% |
| d=4 | 23171.0 | 2035.5 | 1225.0 | -94.71% |

### ABLATION_2 (`ablation_rq1_hybrid_mcts.jsonl`, matches=2760)

| Candidate | Budget | Score [95% CI] | Mean (ms) | Avg Sims | Cache Hit |
|:---|:---:|:---:|---:|---:|---:|
| `Policy_Fast` | `none=0` | 0.542 [0.496, 0.588] | 0.35 | 0.0 | - |
| `Policy_Strong` | `none=0` | 0.525 [0.479, 0.575] | 0.34 | 0.0 | - |
| `Value_Greedy` | `none=0` | 0.296 [0.225, 0.371] | 0.43 | 0.0 | - |
| `PureMCTS_sim50` | `simulations=50` | 0.529 [0.471, 0.588] | 3.79 | 50.0 | - |
| `HybridRollout_sim50` | `simulations=50` | 0.583 [0.542, 0.625] | 8.78 | 50.0 | 47.0% |
| `HybridMixed_sim50` | `simulations=50` | 0.575 [0.533, 0.621] | 8.99 | 50.0 | 60.7% |
| `PureMCTS_sim150` | `simulations=150` | 0.604 [0.571, 0.642] | 10.75 | 150.0 | - |
| `HybridRollout_sim150` | `simulations=150` | 0.579 [0.537, 0.621] | 22.81 | 150.0 | 47.3% |
| `HybridMixed_sim150` | `simulations=150` | 0.567 [0.521, 0.613] | 23.67 | 150.0 | 63.9% |
| `PureMCTS_sim400` | `simulations=400` | 0.621 [0.583, 0.658] | 28.51 | 400.0 | - |
| `HybridRollout_sim400` | `simulations=400` | 0.558 [0.521, 0.596] | 55.24 | 400.0 | 47.1% |
| `HybridMixed_sim400` | `simulations=400` | 0.600 [0.562, 0.637] | 56.74 | 400.0 | 62.8% |
| `PureMCTS_time10ms` | `time_ms=10.0` | 0.613 [0.571, 0.654] | 9.27 | 190.7 | - |
| `HybridRollout_time10ms` | `time_ms=10.0` | 0.575 [0.533, 0.617] | 9.41 | 144.0 | 51.2% |
| `HybridMixed_time10ms` | `time_ms=10.0` | 0.583 [0.542, 0.625] | 9.47 | 145.9 | 68.7% |
| `PureMCTS_time50ms` | `time_ms=50.0` | 0.608 [0.571, 0.646] | 28.49 | 396.8 | - |
| `HybridRollout_time50ms` | `time_ms=50.0` | 0.583 [0.542, 0.625] | 33.38 | 313.2 | 52.1% |
| `HybridMixed_time50ms` | `time_ms=50.0` | 0.579 [0.533, 0.625] | 34.04 | 312.1 | 69.0% |

### ABLATION_3 (`ablation_rq2_seat_advantage.jsonl`, matches=2080)

| Board | Matches | X Win% | Draw% | O Win% | ΔScore (X-O) [95% CI] |
|:---|---:|---:|---:|---:|:---:|
| `3x3_k3` | 840 | 23.8% | 63.6% | 12.6% | +0.112 [+0.073, +0.152] |
| `4x4_k3` | 400 | 76.5% | 0.0% | 23.5% | +0.530 [+0.445, +0.610] |
| `4x4_k4` | 840 | 14.2% | 74.1% | 11.8% | +0.024 [-0.011, +0.058] |

### ABLATION_4 (`ablation_rq3_rl_generalization.jsonl`, matches=3600)

| Board | Candidate | Group | Overall Score | Unseen MCTS | Unseen All | Mean (ms) |
|:---|:---|:---:|---:|---:|---:|---:|
| `4x4_k4` | `SL_FastPolicy` | Supervised | 0.514 | 0.425 | 0.463 | 0.32 |
| `4x4_k4` | `SL_StrongPolicy` | Supervised | 0.497 | 0.344 | 0.422 | 0.35 |
| `4x4_k4` | `SL_ValueGreedy` | Supervised | 0.307 | 0.256 | 0.253 | 0.45 |
| `4x4_k4` | `RL_DQN_seed42` | RL | 0.582 | 0.537 | 0.519 | 0.26 |
| `4x4_k4` | `RL_DQN_seed123` | RL | 0.621 | 0.575 | 0.537 | 0.23 |
| `4x4_k4` | `RL_DQN_seed999` | RL | 0.556 | 0.550 | 0.525 | 0.23 |
| `3x3_k3` | `SL_FastPolicy` | Supervised | 0.476 | 0.556 | 0.528 | 0.24 |
| `3x3_k3` | `SL_StrongPolicy` | Supervised | 0.485 | 0.581 | 0.541 | 0.25 |
| `3x3_k3` | `SL_ValueGreedy` | Supervised | 0.469 | 0.506 | 0.503 | 0.29 |
| `3x3_k3` | `RL_TabQ_seed42` | RL | 0.653 | 0.594 | 0.547 | 0.03 |
| `3x3_k3` | `RL_TabQ_seed123` | RL | 0.661 | 0.613 | 0.556 | 0.03 |
| `3x3_k3` | `RL_TabQ_seed999` | RL | 0.664 | 0.650 | 0.575 | 0.03 |
