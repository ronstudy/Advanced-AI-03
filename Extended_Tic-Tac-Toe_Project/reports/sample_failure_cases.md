# Sample Failure Modes & Tactical Analysis (Automated from Raw Logs)

All boards and move sequences below are replayed verbatim via `ExtendedTicTacToe` public APIs from raw `logs/*.jsonl` records.

### Case 1 (RQ1 / RQ3): Supervised Policy (`L5_StrongPolicy`) Defeated by Lookahead (`L4_MCTS`, `4x4 K=4`)
- **Context**: Illustrates how zero-search policy imitation misses tactical traps against unseen tree search.
- **Match ID**: `L4_MCTS__vs__L5_StrongPolicy__p0001__g0` (`seed=43`, `budget=default`)
- **Matchup**: `X = L4_MCTS` vs `O = L5_StrongPolicy`
- **Outcome**: **L4_MCTS (X)** in `15` plies (winning line: `[[3, 0], [3, 1], [3, 2], [3, 3]]`)
- **Move Sequence**: 1.X(3,2)[#14] -> 2.O(0,0)[#0] -> 3.X(2,2)[#10] -> 4.O(1,2)[#6] -> 5.X(0,3)[#3] -> 6.O(1,1)[#5] -> 7.X(2,3)[#11] -> 8.O(2,1)[#9] -> 9.X(3,1)[#13] -> 10.O(1,3)[#7] -> 11.X(1,0)[#4] -> 12.O(0,1)[#1] -> 13.X(3,0)[#12] -> 14.O(0,2)[#2] -> 15.X(3,3)[#15]
- **Policy Prior Analysis**: StrongPolicyNet Top-3 Priors before move (0,2)[#2]: (0,2)[#2]: 65.7%, (3,3)[#15]: 31.9%, (2,0)[#8]: 2.4% | Played move prior P(s,a) = 65.7%

```text
Board: 4x4 (K=4) | Result: Player X wins!
     0   1   2   3 
   +---+---+---+---+
 0 | O | O | O | X |
   +---+---+---+---+
 1 | X | O | O | O |
   +---+---+---+---+
 2 | . | O | X | X |
   +---+---+---+---+
 3 |[X]|[X]|[X]|[X]|
   +---+---+---+---+
Winning line: [(3, 0), (3, 1), (3, 2), (3, 3)]
```

### Case 2 (RQ1): `L6_DQN` vs `L7_HybridMCTS` (`4x4 K=4`)
- **Context**: Demonstrates how PUCT search refines learned priors into a tactical victory.
- **Match ID**: `L6_DQN__vs__L7_HybridMCTS__p0000__g0` (`seed=42`, `budget=default`)
- **Matchup**: `X = L6_DQN` vs `O = L7_HybridMCTS`
- **Outcome**: **L7_HybridMCTS (O)** in `12` plies (winning line: `[[0, 0], [0, 1], [0, 2], [0, 3]]`)
- **Move Sequence**: 1.X(1,1)[#5] -> 2.O(1,2)[#6] -> 3.X(2,1)[#9] -> 4.O(2,2)[#10] -> 5.X(3,1)[#13] -> 6.O(0,1)[#1] -> 7.X(1,0)[#4] -> 8.O(0,0)[#0] -> 9.X(3,3)[#15] -> 10.O(0,2)[#2] -> 11.X(3,2)[#14] -> 12.O(0,3)[#3]
- **Policy Prior Analysis**: StrongPolicyNet Top-3 Priors before move (3,2)[#14]: (0,3)[#3]: 88.5%, (3,2)[#14]: 11.4%, (3,0)[#12]: 0.1% | Played move prior P(s,a) = 11.4%

```text
Board: 4x4 (K=4) | Result: Player O wins!
     0   1   2   3 
   +---+---+---+---+
 0 |[O]|[O]|[O]|[O]|
   +---+---+---+---+
 1 | X | X | O | . |
   +---+---+---+---+
 2 | . | X | O | . |
   +---+---+---+---+
 3 | . | X | X | X |
   +---+---+---+---+
Winning line: [(0, 0), (0, 1), (0, 2), (0, 3)]
```

### Case 3 (RQ2): Amplified First-Mover Fork Victory on `4x4, K=3`
- **Context**: Illustrates rapid first-mover (`X`) fork conversion when `K=3` on a `4x4` board.
- **Match ID**: `rq2_4x4_k3__L1_Random__vs__L2_RuleBased__p0000__g1` (`seed=1042`, `budget=rq2_4x4_k3`)
- **Matchup**: `X = L2_RuleBased` vs `O = L1_Random`
- **Outcome**: **L2_RuleBased (X)** in `5` plies (winning line: `[[2, 0], [2, 1], [2, 2]]`)
- **Move Sequence**: 1.X(2,1)[#9] -> 2.O(1,1)[#5] -> 3.X(2,2)[#10] -> 4.O(0,1)[#1] -> 5.X(2,0)[#8]

```text
Board: 4x4 (K=3) | Result: Player X wins!
     0   1   2   3 
   +---+---+---+---+
 0 | . | O | . | . |
   +---+---+---+---+
 1 | . | O | . | . |
   +---+---+---+---+
 2 |[X]|[X]|[X]| . |
   +---+---+---+---+
 3 | . | . | . | . |
   +---+---+---+---+
Winning line: [(2, 0), (2, 1), (2, 2)]
```

### Case 4 (3x3 K=3 Control): `L5_StrongPolicy` Outplayed by `L7_HybridMCTS`
- **Context**: Compares raw Policy prior against Hybrid PUCT outcome on `3x3 K=3`.
- **Match ID**: `L5_StrongPolicy__vs__L7_HybridMCTS__p0000__g1` (`seed=42`, `budget=default`)
- **Matchup**: `X = L7_HybridMCTS` vs `O = L5_StrongPolicy`
- **Outcome**: **L7_HybridMCTS (X)** in `7` plies (winning line: `[[0, 0], [1, 1], [2, 2]]`)
- **Move Sequence**: 1.X(0,0)[#0] -> 2.O(0,1)[#1] -> 3.X(1,0)[#3] -> 4.O(2,0)[#6] -> 5.X(1,1)[#4] -> 6.O(1,2)[#5] -> 7.X(2,2)[#8]
- **Policy Prior Analysis**: StrongPolicyNet Top-3 Priors before move (1,2)[#5]: (1,2)[#5]: 46.5%, (2,2)[#8]: 40.9%, (0,2)[#2]: 7.2% | Played move prior P(s,a) = 46.5%

```text
Board: 3x3 (K=3) | Result: Player X wins!
     0   1   2 
   +---+---+---+
 0 |[X]| O | . |
   +---+---+---+
 1 | X |[X]| O |
   +---+---+---+
 2 | O | . |[X]|
   +---+---+---+
Winning line: [(0, 0), (1, 1), (2, 2)]
```
