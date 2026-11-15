# PROGRESS — Theo dõi Tiến độ, Thực nghiệm & Nhật ký Lỗi Kỹ thuật

Tài liệu cập nhật trạng thái thực tế của dự án **Game AI Đa Cấp Độ (Extended Tic-Tac-Toe `M×N, K`)**. Mọi trạng thái `DONE` đều phải đi kèm bằng chứng kiểm chứng (`pytest` output, đường dẫn checkpoint, hoặc raw log) theo đúng quy định trong [RULES.md](./RULES.md) và kế hoạch tại [TASKS.md](./TASKS.md).

- **Cập nhật lần cuối:** `2026-11-15`
- **Giai đoạn hiện tại:** Hoàn tất **Sprint 2 (`5 / 5`)** — Vượt cổng nghiệm thu Cấp 1–4, sẵn sàng bước vào **Sprint 3 (`T3.1`–`T3.4`)**

---

## 1. Tổng quan Trạng thái Dự án

| Giai đoạn | Mô tả | Tiến độ | Trạng thái | Cổng nghiệm thu |
|---|---|---|---|---|
| **Phase 0** | Bộ 6 tài liệu Planning (`PRD`, `RULES`, `ARCHITECTURE`, `DESIGN`, `TASKS`, `PROGRESS`) | `6 / 6` | `DONE` | Đã duyệt & lưu đủ trong `Planning/` |
| **Sprint 1** | Hạ tầng, Configs, Core `engine` & `test_engine.py` | `4 / 4` | `DONE` | `pytest tests/test_engine.py` pass 100% (`29 passed`); `ruff` sạch |
| **Sprint 2** | Tác tử Cấp 1–4 (`Random`, `RuleBased`, `Minimax/AB`, `MCTS UCT`) | `5 / 5` | `DONE` | `pytest tests/test_agents_contract.py tests/test_search_agents.py` pass 100% (`45 passed`; full suite `216 passed`); `ruff` sạch |
| **Sprint 3** | Tác tử Cấp 5–7 (`Fast/Strong Policy`, `ValueNet`, `TabularQ/DQN`, `HybridMCTS`) | `0 / 4` | `TODO` | Đủ checkpoints + `meta.json` hợp lệ |
| **Sprint 4** | Giải đấu Paired Round-Robin, Metrics & 4 Ablations (`RQ3, RQ4, RQ5`) | `0 / 4` | `TODO` | Raw JSONL logs & 95% CI hoàn chỉnh |
| **Sprint 5** | Rich CLI Demo, Visualizer (`Fig 1..5`), `README.md` & Rà soát Rubric | `0 / 3` | `TODO` | Demo ổn định + tái lập 1-lệnh |

---

## 2. Bảng Theo dõi Chi tiết Theo Task

### Phase 0: Tài liệu Thiết kế & Quy chuẩn
| Mã | Tên tài liệu | Trạng thái | Ngày hoàn thành | Đường dẫn |
|---|---|---|---|---|
| **P0.1** | Đặc tả yêu cầu & 3 Câu hỏi Nghiên cứu (`RQ3, RQ4, RQ5`) | `DONE` | `2026-11-14` | [PRD.md](./PRD.md) |
| **P0.2** | Quy chuẩn kỹ thuật & Trung thực học thuật | `DONE` | `2026-11-14` | [RULES.md](./RULES.md) |
| **P0.3** | Kiến trúc phân lớp, API, NN/RL/Hybrid & Schemas | `DONE` | `2026-11-14` | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| **P0.4** | Thiết kế giao diện `Rich CLI` & 5 bộ biểu đồ báo cáo | `DONE` | `2026-11-14` | [DESIGN.md](./DESIGN.md) |
| **P0.5** | Kế hoạch phân rã công việc theo 5 Sprint | `DONE` | `2026-11-14` | [TASKS.md](./TASKS.md) |
| **P0.6** | Bảng theo dõi tiến độ & nhật ký lỗi | `DONE` | `2026-11-14` | [PROGRESS.md](./PROGRESS.md) |

### Sprint 1 – Sprint 5: Thực thi & Kiểm thử
| Mã | Nội dung tóm tắt | Trạng thái | Ngày cập nhật | Bằng chứng kiểm chứng / Ghi chú |
|---|---|---|---|---|
| **T1.1** | Khởi tạo `pyproject.toml`, `configs/*.yaml`, `config.py`, `seed_everything()` | `DONE` | `2026-11-14` | `pip install --no-build-isolation --no-deps -e .` OK; `pytest tests/test_config.py -v`: `24 passed`; `ruff check src/ tests/test_config.py`: `All checks passed!` |
| **T1.2** | `engine/exceptions.py` & `engine/rules.py` ($O(K)$ win check, canonical, weights) | `DONE` | `2026-11-14` | `pytest tests/test_config.py tests/test_rules.py -v`: `51 passed` (27 tests trong `test_rules.py`); `ruff check src/ tests/`: `All checks passed!` |
| **T1.3** | `ExtendedTicTacToe` (7+1 API chuẩn) trong `engine/game.py` | `DONE` | `2026-11-14` | `pytest -v`: `65 passed` (14 tests trong `test_engine.py`); `ruff check src/ tests/`: `All checks passed!` |
| **T1.4** | Bộ kiểm thử `tests/test_engine.py` (`3×3`, `4×4 K=3/4`, `5×5`) | `DONE` | `2026-11-14` | `pytest tests/test_engine.py -v`: `29 passed`; `pytest -v`: `80 passed`; `ruff check src/ tests/`: `All checks passed!` |
| **T2.1** | `agents/base.py` (`BaseAgent`, `AgentDecision`, `SearchTimeoutError`) | `DONE` | `2026-11-15` | `pytest tests/test_base_agent.py -v`: `27 passed`; `pytest -v`: `107 passed`; `ruff check src/ tests/`: `All checks passed!` |
| **T2.2** | Cấp 1 (`RandomAgent`) & Cấp 2 (`RuleBasedAgent` 6 tầng luật) | `DONE` | `2026-11-15` | `pytest tests/test_level1_level2_agents.py -v`: `16 passed`; `pytest -v`: `123 passed`; `ruff check src/ tests/`: `All checks passed!`; bất bại vét cạn 100% trên `3×3 K=3` (`X` & `O`); đối kháng `4×4 K=4` (`N=100`) đạt `score=0.935` (`87-13-0`, `0.0627 ms/move`) trước `RandomAgent` |
| **T2.3** | Cấp 3 (`MinimaxAgent` / `AlphaBetaAgent`, heuristic, Iterative Deepening) | `DONE` | `2026-11-15` | `pytest tests/test_minimax_agent.py -v`: `14 passed`; `pytest -v`: `137 passed`; `ruff check src/ tests/`: `All checks passed!`; trên `4×4 K=4` (`d=4`), `AlphaBeta + Order` giảm từ `47,297` xuống `1,941` nút (`-95.9%`) mà vẫn giữ nguyên `action=0`, `val=-0.008000` |
| **T2.4** | Cấp 4 (`MCTSAgent` UCT 4 pha) | `DONE` | `2026-11-15` | Hoàn thiện `MCTSAgent` UCT 4 pha (`tests/test_mcts_agent.py`: 34 passed; full suite 171 passed; `ruff` clean); benchmark đối đầu cân bằng `4×4 K=4` đạt `1.000` vs `RandomAgent` và `0.500` (`0-40-0`, 0 thua) vs `RuleBased`/`AlphaBeta(d=4)` |
| **T2.5** | `tests/test_agents_contract.py` & `tests/test_search_agents.py` | `DONE` | `2026-11-15` | `pytest tests/test_agents_contract.py tests/test_search_agents.py -v`: `45 passed`; `pytest -v`: `216 passed in 7.09s`; `ruff check src/ tests/`: `All checks passed!`; AB+Order cắt >85% nút trên `4×4 K=4 d=4`; Round-Robin 4 cấp (120 ván): Cấp 2/3/4 bất bại (`0` thua) |
| **T3.1** | `nn_models.py` (`Fast/StrongPolicyNet`, `ValueNet`, `DQNNet`) & `checkpoint_io.py` | `TODO` | — | — |
| **T3.2** | `dataset_gen.py` (game-level split) & `train_supervised.py` + `policy_agent.py` | `TODO` | — | — |
| **T3.3** | `rl_agent.py` (`TabularQAgent` & `DQNAgent`) + train đa seed (3 seeds) | `TODO` | — | — |
| **T3.4** | Cấp 7 `HybridMCTSAgent` (PUCT + prior cache + `rollout`/`mixed` leaf eval) | `TODO` | — | — |
| **T4.1** | `TournamentRunner` (Paired Round-Robin 50/50 `X/O`, JSONL logger) | `TODO` | — | — |
| **T4.2** | `metrics.py` (Score, Wilson/Bootstrap 95% CI, p95) & `test_tournament_metrics.py` | `TODO` | — | — |
| **T4.3** | Giải đấu chính `4×4 K=4` (Fixed Budget + Default) & đối chứng `3×3 K=3` | `TODO` | — | — |
| **T4.4** | Chạy 4 kịch bản `ablations.py` (AB pruning, **RQ3**, **RQ4**, **RQ5**) | `TODO` | — | — |
| **T5.1** | Giao diện `Rich CLI` (`theme.py`, `cli_app.py`, `scripts/play.py`) | `TODO` | — | — |
| **T5.2** | `visualizer.py` & `plot_results.py` (`Fig 1..5`, `summary_tables.md`, `sample_failure_cases.md`) | `TODO` | — | — |
| **T5.3** | `README.md`, đối chiếu trích dẫn `References/` & kiểm tra chéo Rubric 100đ | `TODO` | — | — |

---

## 3. Nhật ký Huấn luyện & Thực nghiệm (`Experiment & Checkpoint Registry`)

*(Chỉ ghi nhận các lần chạy đã thực thi hoàn chỉnh kèm đường dẫn log/checkpoint thực tế; tuyệt đối không điền trước số liệu dự đoán.)*

| Run ID | Ngày | Loại chạy (`Train` / `Tournament` / `Ablation`) | Cấu hình & Seed | Kết quả tóm tắt (Loss / Acc / Score) | Đường dẫn Checkpoint / Raw Log |
|---|---|---|---|---|---|
| `T2.2-SANITY-01` | `2026-11-15` | `Benchmark (Paired)` | `3×3 K=3`, `RuleBased` (`seed=42..91`) vs `Random` (`seed=1000..1049`), `N=100` (50 cặp `X/O`) | `W-D-L = 91-9-0`, `score = 0.955`; Rule `mean = 0.0508 ms` (`p95 = 0.0621 ms`, `avg_nodes = 18.79`); Random `mean = 0.0228 ms` (`p95 = 0.0281 ms`) | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.2-SANITY-02` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `RuleBased` (`seed=42..91`) vs `Random` (`seed=1000..1049`), `N=100` (50 cặp `X/O`) | `W-D-L = 87-13-0`, `score = 0.935`; Rule `mean = 0.0627 ms` (`p95 = 0.0847 ms`, `avg_nodes = 32.30`); Random `mean = 0.0255 ms` (`p95 = 0.0309 ms`) | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.2-SANITY-03` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=3`, `RuleBased` (`seed=42..91`) vs `Random` (`seed=1000..1049`), `N=100` (50 cặp `X/O`) | `W-D-L = 99-0-1`, `score = 0.990`; Rule `mean = 0.0862 ms` (`p95 = 0.1094 ms`, `avg_nodes = 42.27`); Random `mean = 0.0260 ms` (`p95 = 0.0302 ms`) | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.3-SANITY-01` | `2026-11-15` | `Ablation Preview` | `4×4 K=4` (bàn rỗng), `Minimax d=4`, bật/tắt `AB` & `MoveOrdering` | `Minimax`: `47,297` nút (`523.158 ms`); `AB` không order: `3,169` nút (`89.170 ms`); `AB + Order`: `1,941` nút (`50.673 ms`, `-95.9%` nút); tất cả đều trả `action=0`, `val=-0.008000` | Inline sanity run (log Ablation 1 chính thức ở Sprint 4) |
| `T2.3-SANITY-02` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `AlphaBeta(d=4, 50ms)` vs `RandomAgent(seed=100..119)`, `N=40` (20 cặp `X/O`) | `W-D-L = 28-12-0`, `score = 0.850`; `mean = 29.202 ms`, `p95 = 56.036 ms`, `avg_nodes = 1060.9`, `avg_depth = 3.28` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.3-SANITY-03` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `AlphaBeta(d=4, 50ms)` vs `RuleBased(seed=100..119)`, `N=40` (20 cặp `X/O`) | `W-D-L = 0-40-0`, `score = 0.500`; `mean = 23.044 ms`, `p95 = 55.713 ms`, `avg_nodes = 886.7`, `avg_depth = 3.44` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.4-SANITY-01` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `MCTS(n=50)` vs `RandomAgent`, `N=40` (20 cặp `X/O`) | `W-D-L = 35-5-0`, `score = 0.938`; `mean = 4.473 ms`, `p95 = 7.607 ms`, `avg_sims = 50.0`, `avg_nodes = 45.3` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.4-SANITY-02` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `MCTS(n=200)` vs `RandomAgent`, `N=40` (20 cặp `X/O`) | `W-D-L = 40-0-0`, `score = 1.000`; `mean = 42.550 ms`, `p95 = 84.315 ms`, `avg_sims = 200.0`, `avg_nodes = 175.8` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.4-SANITY-03` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `MCTS(n=200)` vs `RuleBased`, `N=40` (20 cặp `X/O`) | `W-D-L = 0-40-0`, `score = 0.500`; `mean = 38.167 ms`, `p95 = 78.275 ms`, `avg_sims = 200.0`, `avg_nodes = 152.0` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.4-SANITY-04` | `2026-11-15` | `Benchmark (Paired)` | `4×4 K=4`, `MCTS(n=200, 50ms)` vs `AlphaBeta(d=4, 50ms)`, `N=40` (20 cặp `X/O`) | `W-D-L = 0-40-0`, `score = 0.500`; `mean = 29.724 ms`, `p95 = 50.418 ms`, `avg_sims = 185.9`, `avg_nodes = 139.0` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.4-SANITY-05` | `2026-11-15` | `Benchmark (Paired)` | `3×3 K=3`, `MCTS(n=200)` vs `RandomAgent`, `N=40` (20 cặp `X/O`) | `W-D-L = 39-1-0`, `score = 0.988`; `mean = 21.317 ms`, `p95 = 40.098 ms`, `avg_sims = 200.0`, `avg_nodes = 127.9` | Inline sanity run (chưa xuất `.jsonl`; log chính thức ở Sprint 4) |
| `T2.5-SPRINT2-RR` | `2026-11-15` | `Tournament (Paired RR)` | `4×4 K=4`, 4 cấp (`L1 Random`, `L2 RuleBased`, `L3 AB d=4/50ms`, `L4 MCTS 200/50ms`), 10 cặp/cặp đấu (`N=60` ván/tác tử, `120` ván tổng) | `L1`: `0-9-51` (`0.075`, `0.028 ms`, `p95=0.037`); `L2`: `18-42-0` (`0.650`, `0.065 ms`, `p95=0.093`); `L3`: `15-45-0` (`0.625`, `10.706 ms`, `p95=26.465`); `L4`: `18-42-0` (`0.650`, `15.857 ms`, `p95=29.021`); mọi cặp đấu giữa `L2/L3/L4` đều `0-20-0` (`0.500`) | Inline Sprint 2 gate benchmark (log `.jsonl` chính thức ở Sprint 4) |

---

## 4. Nhật ký Lỗi Kỹ thuật (`Bug & Issue Log`)

| Bug ID | Ngày phát hiện | Module | Triệu chứng & Nguyên nhân gốc (*Root Cause*) | Cách khắc phục | Trạng thái |
|---|---|---|---|---|---|
| *(Chưa có)* | — | — | — | — | — |

---

## 5. Hành động Tiếp theo (`Next Actions`)

1. Khởi động **Sprint 3 (`T3.1` – `T3.4`) — Tác tử Học sâu, Học tăng cường & Mô hình Lai (Cấp 5–7)**:
   - **T3.1:** Cài đặt `nn_models.py` (`FastPolicyNet`, `StrongPolicyNet`, `ValueNet`, `DQNNet` hỗ trợ `(B, 1, M, N)`), cơ chế Action Masking dùng chung và `checkpoint_io.py` kèm xác minh `*.meta.json`.
   - **T3.2:** Xây dựng `dataset_gen.py` (chia `70/15/15` theo `game_id`), `train_supervised.py` và `policy_agent.py` (Cấp 5).
   - **T3.3:** Xây dựng & huấn luyện đa seed `TabularQAgent` (`3×3 K=3`) và `DQNAgent` Self-Play (`4×4 K=4`, `4×4 K=3`) trong `rl_agent.py` (Cấp 6).
   - **T3.4:** Cài đặt `HybridMCTSAgent` (PUCT + prior cache + chế độ `rollout` / `mixed`) (Cấp 7).
2. Vượt qua cổng nghiệm thu Sprint 3 (đủ checkpoints thực tế kèm metadata + `test_agents_contract.py` pass 100% cho đủ 7 cấp độ) trước khi sang Sprint 4.
