# PROGRESS — Theo dõi Tiến độ, Thực nghiệm & Nhật ký Lỗi Kỹ thuật

Tài liệu cập nhật trạng thái thực tế của dự án **Game AI Đa Cấp Độ (Extended Tic-Tac-Toe `M×N, K`)**. Mọi trạng thái `DONE` đều phải đi kèm bằng chứng kiểm chứng (`pytest` output, đường dẫn checkpoint, hoặc raw log) theo đúng quy định trong [RULES.md](./RULES.md) và kế hoạch tại [TASKS.md](./TASKS.md).

- **Cập nhật lần cuối:** `2026-11-14`
- **Giai đoạn hiện tại:** Đang thực hiện **Sprint 1** (Hoàn thành `T1.1`–`T1.3`, tiếp theo `T1.4`)

---

## 1. Tổng quan Trạng thái Dự án

| Giai đoạn | Mô tả | Tiến độ | Trạng thái | Cổng nghiệm thu |
|---|---|---|---|---|
| **Phase 0** | Bộ 6 tài liệu Planning (`PRD`, `RULES`, `ARCHITECTURE`, `DESIGN`, `TASKS`, `PROGRESS`) | `6 / 6` | `DONE` | Đã duyệt & lưu đủ trong `Planning/` |
| **Sprint 1** | Hạ tầng, Configs, Core `engine` & `test_engine.py` | `3 / 4` | `IN_PROGRESS` | `pytest tests/test_engine.py` pass 100% |
| **Sprint 2** | Tác tử Cấp 1–4 (`Random`, `RuleBased`, `Minimax/AB`, `MCTS UCT`) | `0 / 5` | `TODO` | Contract & Search tests pass 100% |
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
| **T1.4** | Bộ kiểm thử `tests/test_engine.py` (`3×3`, `4×4 K=3/4`, `5×5`) | `TODO` | — | — |
| **T2.1** | `agents/base.py` (`BaseAgent`, `AgentDecision`, `SearchTimeoutError`) | `TODO` | — | — |
| **T2.2** | Cấp 1 (`RandomAgent`) & Cấp 2 (`RuleBasedAgent` 6 tầng luật) | `TODO` | — | — |
| **T2.3** | Cấp 3 (`MinimaxAgent` / `AlphaBetaAgent`, heuristic, Iterative Deepening) | `TODO` | — | — |
| **T2.4** | Cấp 4 (`MCTSAgent` UCT 4 pha) | `TODO` | — | — |
| **T2.5** | `tests/test_agents_contract.py` & `tests/test_search_agents.py` | `TODO` | — | — |
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
| *(Chưa có)* | — | — | — | — | — |

---

## 4. Nhật ký Lỗi Kỹ thuật (`Bug & Issue Log`)

| Bug ID | Ngày phát hiện | Module | Triệu chứng & Nguyên nhân gốc (*Root Cause*) | Cách khắc phục | Trạng thái |
|---|---|---|---|---|---|
| *(Chưa có)* | — | — | — | — | — |

---

## 5. Hành động Tiếp theo (`Next Actions`)

1. Hoàn tất **Sprint 1 (`T1.4`)**:
   - **T1.4:** Mở rộng, rà soát toàn diện `tests/test_engine.py` (`3×3`, `4×4 K=3/4`, `5×5`) và chạy cổng nghiệm thu Sprint 1 (`pytest tests/test_engine.py` pass 100%, `ruff check src/ttt_ai/engine` sạch lỗi).
2. Sau khi vượt qua cổng nghiệm thu Sprint 1, cập nhật trạng thái `T1.4` sang `DONE` và chuyển Sprint 1 sang `DONE` trước khi sang Sprint 2 (`T2.1`–`T2.5`).
