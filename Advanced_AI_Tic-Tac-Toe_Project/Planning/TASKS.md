# TASKS — Kế hoạch Triển khai Dự án Theo Sprint

Tài liệu này phân rã toàn bộ khối lượng công việc của dự án **Game AI Đa Cấp Độ (Extended Tic-Tac-Toe `M×N, K`)** thành **5 Sprint** tuần tự, bám sát kiến trúc trong [ARCHITECTURE.md](./ARCHITECTURE.md), thiết kế giao diện trong [DESIGN.md](./DESIGN.md), các yêu cầu nghiên cứu (**RQ3, RQ4, RQ5**) trong [PRD.md](./PRD.md) và quy chuẩn kiểm soát chất lượng trong [RULES.md](./RULES.md).

---

## Sprint 1: Khởi tạo Hạ tầng, Lõi Trò chơi (`engine`) & Bộ Kiểm thử Đơn vị

**Mục tiêu Sprint:** Hoàn thiện bộ khung package `ttt_ai`, cấu hình YAML chuẩn hóa, lõi trò chơi `ExtendedTicTacToe` tổng quát cho mọi `M×N, K` và đạt 100% pass `pytest` cho Engine trước khi xây dựng bất kỳ tác tử nào.

| Mã | Công việc chi tiết | Tệp / Module mục tiêu | Tiêu chí hoàn thành (DoD) |
|---|---|---|---|
| **T1.1** | Khởi tạo cấu trúc gói Python (`src/` layout), `pyproject.toml` (dependencies: `numpy`, `torch`, `pyyaml`, `rich`, `matplotlib`, `pytest`, `ruff`), tạo bộ cấu hình chuẩn (`4×4 K=4`, `4×4 K=3`, `3×3 K=3`, `5×5 K=4`) và hàm `seed_everything()`. | `pyproject.toml`, `configs/game_*.yaml`, `src/ttt_ai/config.py` | `pip install -e .` thành công; load YAML ra Typed Dataclasses không lỗi; cố định được seed cho `random`, `numpy`, `torch`. |
| **T1.2** | Cài đặt ngoại lệ `IllegalMoveError`, `InvalidConfigError` và các hàm luật dùng chung: `check_win_around()` ($O(K)$ theo 4 trục), `precompute_win_windows()`, `positional_weights()`, `encode_canonical_board()`. | `src/ttt_ai/engine/exceptions.py`, `src/ttt_ai/engine/rules.py` | Không phụ thuộc `torch`/`rich`; `check_win_around` trả về đúng danh sách tọa độ `K` ô thắng hoặc `None`. |
| **T1.3** | Xây dựng lớp `ExtendedTicTacToe` cài đặt đầy đủ Game API chuẩn: `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result(player)`, `render()`, `clone()`, `undo()`. | `src/ttt_ai/engine/game.py` | `step()` ném `IllegalMoveError` khi đánh sai; `undo()` khôi phục trạng thái $O(1)$; `clone()` độc lập hoàn toàn về bộ nhớ. |
| **T1.4** | Viết bộ unit test toàn diện cho Engine trên nhiều cấu hình (`3×3 K=3`, `4×4 K=3`, `4×4 K=4`, `5×5 K=4`): thắng ngang/dọc/chéo/biên, hòa đầy bàn, ngoại lệ, `clone()`, `undo()`. | `tests/test_engine.py` | `pytest tests/test_engine.py` pass 100% (0 skipped/failed). |

**Cổng nghiệm thu Sprint 1:** `pytest tests/test_engine.py` pass 100%; `ruff check src/ttt_ai/engine` sạch lỗi PEP 8 và Type Hints.

---

## Sprint 2: Tác tử Baseline & Tìm kiếm Cổ điển (Cấp 1 – 4)

**Mục tiêu Sprint:** Hoàn thiện tầng giao ước `BaseAgent` / `AgentDecision` và 4 cấp tác tử bắt buộc (`Random`, `RuleBased`, `Minimax/AlphaBeta`, `MCTS UCT`) kèm kiểm thử hợp đồng API.

| Mã | Công việc chi tiết | Tệp / Module mục tiêu | Tiêu chí hoàn thành (DoD) |
|---|---|---|---|
| **T2.1** | Xây dựng `BaseAgent`, `AgentDecision` (chứa `action`, `latency_ms`, `nodes_visited`, `simulations`, `depth_reached`, `value_estimate`, `action_scores`, `prior_probs`) và `SearchTimeoutError`. | `src/ttt_ai/agents/base.py` | Đo `latency_ms` bằng `time.perf_counter()` bao trùm toàn bộ `select_action()`. |
| **T2.2** | Cài đặt **Cấp 1 (`RandomAgent`)** (dùng `np.random.Generator` riêng) và **Cấp 2 (`RuleBasedAgent`)** tổng quát hóa 6 tầng luật ưu tiên Newell & Simon (Thắng ngay $\rightarrow$ Chặn ngay $\rightarrow$ Tạo fork $\rightarrow$ Chặn fork $\rightarrow$ `positional_weights` $\rightarrow$ Tie-break). | `src/ttt_ai/agents/random_agent.py`, `src/ttt_ai/agents/rule_based_agent.py` | Đánh thắng ngay khi có cơ hội; chặn được nước thắng 1 bước; xử lý xung đột luật nhất quán trên `3×3` và `4×4`. |
| **T2.3** | Cài đặt **Cấp 3 (`MinimaxAgent` / `AlphaBetaAgent`)**: `step()/undo()` trên bản `clone()`, hàm lượng giá `evaluate_board` từ `precompute_win_windows`, cờ `use_alpha_beta` & `use_move_ordering`, và Iterative Deepening theo `time_budget_ms`. | `src/ttt_ai/agents/minimax_agent.py` | Tôn trọng `time_budget_ms` nhờ kiểm tra deadline nội bộ; khi tắt/bật Alpha-Beta ở cùng `max_depth` phải trả về cùng giá trị gốc. |
| **T2.4** | Cài đặt **Cấp 4 (`MCTSAgent`)** chuẩn 4 pha Selection (UCT) – Expansion – Simulation (rollout ngẫu nhiên) – Backpropagation; hỗ trợ dừng theo `n_simulations` hoặc `time_budget_ms`. | `src/ttt_ai/agents/mcts_agent.py` | Trả về `action_scores` theo phân phối visit `N(s,a)/N(s)`; giải phóng cây sau khi chọn nước. |
| **T2.5** | Viết test hợp đồng cho tất cả Agent Cấp 1–4: không làm biến đổi trạng thái bàn cờ gốc, luôn trả nước hợp lệ, kiểm chứng `Minimax == AlphaBeta` trên cây cố định. | `tests/test_agents_contract.py`, `tests/test_search_agents.py` | `pytest` pass 100% trên cả `3×3` và `4×4`. |

**Cổng nghiệm thu Sprint 2:** Toàn bộ tác tử Cấp 1–4 vượt qua `test_agents_contract.py` và `test_search_agents.py`; `AlphaBeta` giảm rõ rệt số nút duyệt so với `Minimax` thuần.

---

## Sprint 3: Tác tử Học sâu, Học tăng cường & Mô hình Lai (Cấp 5 – 7)

**Mục tiêu Sprint:** Xây dựng đầy đủ kiến trúc mạng PyTorch (`FastPolicyNet`, `StrongPolicyNet`, `ValueNet`, `DQNNet`), quy trình sinh dữ liệu sạch chống rò rỉ, huấn luyện đa seed và tích hợp `HybridMCTSAgent` (AlphaGo-inspired).

| Mã | Công việc chi tiết | Tệp / Module mục tiêu | Tiêu chí hoàn thành (DoD) |
|---|---|---|---|
| **T3.1** | Cài đặt `nn_models.py` (`FastPolicyNet`, `StrongPolicyNet`, `ValueNet`, `DQNNet` tương thích `(B, 1, M, N)`), cơ chế Action Masking dùng chung, và `checkpoint_io.py` (lưu/nạp `.pt/.pkl` kèm `*.meta.json` kiểm tra `board_config` & hash). | `src/ttt_ai/agents/nn_models.py`, `src/ttt_ai/training/checkpoint_io.py` | Không cho phép load checkpoint sai kích thước bàn hoặc thiếu metadata đã huấn luyện. |
| **T3.2** | Xây dựng `dataset_gen.py` (sinh ván expert từ `AlphaBeta`/`RuleBased` có nhiễu mở đầu, ghi `game_id`, tách `train/val/test` 70/15/15 **theo `game_id`**) và `train_supervised.py` huấn luyện `FastPolicyNet`, `StrongPolicyNet`, `ValueNet` cho `4×4 K=4` (và `3×3 K=3`). | `src/ttt_ai/training/dataset_gen.py`, `src/ttt_ai/training/train_supervised.py`, `src/ttt_ai/agents/policy_agent.py`, `scripts/generate_dataset.py` | Lưu checkpoint tốt nhất theo `val_loss`; ghi log learning curve (loss/acc theo epoch); `PolicyNetworkAgent` & `ValueGreedyAgent` suy luận có Action Masking. |
| **T3.3** | Cài đặt và huấn luyện **Cấp 6 (`TabularQAgent` cho `3×3 K=3` & `DQNAgent` Self-Play + Mixed Opponent Pool cho `4×4 K=4`, `4×4 K=3`)** với Experience Replay, Target Network và zero-sum Bellman target, chạy qua **3 `random seed`**. | `src/ttt_ai/agents/rl_agent.py`, `src/ttt_ai/training/train_q_tabular.py`, `src/ttt_ai/training/train_dqn_selfplay.py`, `scripts/train_all.py` | Lưu đủ 3 checkpoint & learning curves đa seed; mô hình thắng áp đảo `RandomAgent` (> 85% trên `4×4`). |
| **T3.4** | Cài đặt **Cấp 7 (`HybridMCTSAgent`)** dùng PUCT với prior $P(s, a)$ từ `StrongPolicyNet`, **LRU Cache theo hash bàn cờ**, hỗ trợ 2 chế độ đánh giá lá (`rollout` và `mixed` kết hợp `ValueNet`). | `src/ttt_ai/agents/hybrid_mcts_agent.py` | Ghi nhận đồng thời `prior_probs` và `action_scores` trong `AgentDecision`; pass contract tests (`clone`, `action_masking`). |

**Cổng nghiệm thu Sprint 3:** Đủ checkpoint đã huấn luyện thực tế + metadata cho Cấp 5, 6, 7; `test_agents_contract.py` pass 100% cho cả 7 cấp độ.

---

## Sprint 4: Hệ thống Giải đấu Công bằng, Thống kê & 4 Thí nghiệm Ablation

**Mục tiêu Sprint:** Tổ chức giải đấu Round-Robin cân bằng lượt đi trước/sau, tính độ đo thống kê kèm khoảng tin cậy 95%, và thu thập đủ dữ liệu thực nghiệm trả lời **RQ3, RQ4, RQ5**.

| Mã | Công việc chi tiết | Tệp / Module mục tiêu | Tiêu chí hoàn thành (DoD) |
|---|---|---|---|
| **T4.1** | Xây dựng `TournamentRunner` chạy **Paired Round-Robin** (chia đều 50% `X` / 50% `O` trên cùng cặp seed), tự động ghi metadata phần cứng + từng nước đi vào `logs/*.jsonl`. | `src/ttt_ai/benchmark/tournament.py`, `configs/tournament.yaml`, `scripts/run_tournament.py` | Log JSONL đầy đủ `meta` header và từng `match` record; hỗ trợ chạy tiếp hoặc tái lập theo seed. |
| **T4.2** | Cài đặt `metrics.py`: tính `score = (W + 0.5*D)/N`, **Wilson Score 95% CI**, **Bootstrap 95% CI**, $\Delta\text{Score}_X - \text{Score}_O$, `mean_ms`, `p95_ms`, `avg_nodes/sims` + unit tests. | `src/ttt_ai/benchmark/metrics.py`, `tests/test_tournament_metrics.py` | `pytest tests/test_tournament_metrics.py` pass 100%. |
| **T4.3** | Chạy giải đấu chính thức trên **`4×4, K=4`** (200 ván/cặp ở cả 2 chế độ: *Fixed Time Budget* `10 ms / 50 ms` và *Default Config*) + giải đấu đối chứng trên **`3×3, K=3`**. | `logs/tournament_4x4_k4_*.jsonl`, `logs/tournament_3x3_k3_*.jsonl` | Không có ván nào bị lỗi `IllegalMoveError`; đầy đủ số ván cho mọi cặp trong 7 cấp AI. |
| **T4.4** | Cài đặt & chạy 4 kịch bản trong `ablations.py`: (1) Alpha-Beta bật/tắt cắt tỉa & move ordering; (2) **RQ3** `Policy-only` vs `Pure MCTS` vs `Hybrid MCTS (rollout & mixed)` theo `n_simulations ∈ {50, 150, 400}` & `time_budget_ms`; (3) **RQ4** $\Delta\text{Score}$ trên `3×3 K=3`, `4×4 K=3`, `4×4 K=4`; (4) **RQ5** `RL Self-play` vs `Supervised` qua 3 seed trước unseen opponents. | `src/ttt_ai/benchmark/ablations.py`, `scripts/run_ablations.py`, `logs/ablation_*.jsonl` | Có raw JSONL log riêng cho cả 4 ablation phục vụ vẽ biểu đồ tự động. |

**Cổng nghiệm thu Sprint 4:** Toàn bộ raw log giải đấu & 4 ablations hoàn chỉnh, không chỉnh sửa tay, sẵn sàng tổng hợp số liệu trả lời RQ3, RQ4, RQ5.

---

## Sprint 5: Giao diện Rich CLI, Trực quan hóa Báo cáo & Hoàn thiện Tài liệu

**Mục tiêu Sprint:** Hoàn thiện demo `Rich CLI` ổn định ở cả chế độ Người–Máy và Máy–Máy, tự động sinh toàn bộ biểu đồ/bảng số liệu báo cáo từ raw log, và kiểm tra chéo với Rubric môn học.

| Mã | Công việc chi tiết | Tệp / Module mục tiêu | Tiêu chí hoàn thành (DoD) |
|---|---|---|---|
| **T5.1** | Xây dựng `theme.py` (bảng màu Rich, thang nhiệt heatmap 5 mức, đường thắng nổi bật, fallback ASCII) và `cli_app.py` (Setup Wizard + Live Match Dashboard: Board, AI Telemetry, Action Heatmap & Top-5 Moves, hỗ trợ `undo`, `help`, `quit`). | `src/ttt_ai/ui/theme.py`, `src/ttt_ai/ui/cli_app.py`, `scripts/play.py` | Chơi thử mượt cả `Human vs AI` và `AI vs AI` trên Windows Terminal/PowerShell; nhập sai tọa độ không crash; Cấp 7 hiển thị cả `prior_probs` và `action_scores`. |
| **T5.2** | Xây dựng `visualizer.py` & `plot_results.py` đọc trực tiếp từ `logs/*.jsonl` để sinh **Fig 1..5** (PNG 300 DPI), `reports/summary_metrics.csv`, `reports/summary_tables.md`, và `reports/sample_failure_cases.md`. | `src/ttt_ai/benchmark/visualizer.py`, `scripts/plot_results.py`, `reports/` | 100% số liệu trong bảng khớp với raw JSONL logs; hình ảnh rõ nét, không đè chữ. |
| **T5.3** | Viết `README.md` (hướng dẫn cài đặt, chạy test, sinh dữ liệu, train, chạy giải đấu/ablation, vẽ biểu đồ, chạy demo Rich CLI), đối chiếu các file tham khảo trong `References/` để ghi rõ phần kế thừa/tự cài đặt, và rà soát checklist Rubric 100 điểm + Bonus. | `README.md`, `Planning/PROGRESS.md` | Một người mới clone repo có thể chạy lại test, plot và demo Rich CLI theo đúng hướng dẫn trong `README.md`. |

**Cổng nghiệm thu Sprint 5:** Demo `play.py` chạy ổn định; toàn bộ `pytest` pass; bộ 5 hình + bảng tổng hợp + báo cáo sẵn sàng đóng gói nộp bài.
