# Xây dựng và Đánh giá Game AI Đa Cấp Độ — Extended Tic-Tac-Toe (`M×N, K`)

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1.0-ee4c2c.svg)](./pyproject.toml)
[![Pytest](https://img.shields.io/badge/Pytest-347%20passed-brightgreen.svg)](#6-huong-dan-tai-lap-thuc-nghiem-1-lenh)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20%7C%20PEP%208-000000.svg)](./pyproject.toml)
[![Deliverables](https://img.shields.io/badge/Deliverables-LaTeX%20%7C%20Slide%20%7C%20Poster%20%7C%20Notebook-orange.svg)](#4-bo-san-pham-bao-cao-thuyet-trinh--so-tay-thuc-nghiem-reports_project--jupiter_notebook)
[![Academic Integrity](https://img.shields.io/badge/Integrity-Zero%20Fabrication-informational.svg)](#8-minh-bach-hoc-thuat--ke-thua)

> **Hệ thống mô phỏng môi trường đối kháng, huấn luyện, giải đấu công bằng và trình diễn Rich CLI cho bài toán Game AI 7 cấp độ trên trò chơi Extended Tic-Tac-Toe tổng quát `M×N, K` (chuẩn thực nghiệm chính `4×4, K=4`).**

---

## Thông tin chung

| Mục | Chi tiết |
|---|---|
| **Học phần** | Trí Tuệ Nhân Tạo Nâng Cao (`841458`) |
| **Giảng viên hướng dẫn** | Đỗ Như Tài |
| **Nhóm** | 03 |
| **Sinh viên thực hiện** | Trương Chí Long |
| **MSSV** | `3123410198` |
| **Đơn vị** | Trường Đại học Sài Gòn (SGU) |
| **Repository** | [github.com/ronstudy/Advanced-AI-03](https://github.com/ronstudy/Advanced-AI-03) |

---

## Mục lục

1. [Tổng quan Dự án & Ba Câu hỏi Nghiên cứu (`RQ1, RQ2, RQ3`)](#1-tong-quan-du-an--ba-cau-hoi-nghien-cuu-rq1-rq2-rq3)
2. [Kiến trúc Hệ thống & 7 Cấp độ Tác tử AI](#2-kien-truc-he-thong--7-cap-do-tac-tu-ai)
3. [Bộ Tài liệu Sổ tay (`Planning/`) & Cấu trúc Thư mục Dự án](#3-bo-tai-lieu-so-tay-planning--cau-truc-thu-muc-du-an)
4. [Bộ Sản phẩm Báo cáo, Thuyết trình & Sổ tay Thực nghiệm (`Reports_Project/` & `Jupiter_Notebook/`)](#4-bo-san-pham-bao-cao-thuyet-trinh--so-tay-thuc-nghiem-reports_project--jupiter_notebook)
5. [Hướng dẫn Cài đặt & Khởi tạo Môi trường (Quick Start)](#5-huong-dan-cai-dat--khoi-tao-moi-truong-quick-start)
6. [Hướng dẫn Tái lập Thực nghiệm 1 Lệnh](#6-huong-dan-tai-lap-thuc-nghiem-1-lenh)
7. [Hướng dẫn Trải nghiệm Trực quan (`Rich CLI` Dashboard)](#7-huong-dan-trai-nghiem-truc-quan-rich-cli-dashboard)
8. [Minh bạch Học thuật & Kế thừa (`References/`)](#8-minh-bach-hoc-thuat--ke-thua)
9. [Bảng Tự Đánh giá Ánh xạ Rubric (100 Điểm + 10 Điểm Khuyến khích)](#9-bang-tu-danh-gia-anh-xa-rubric-100-diem--10-diem-khuyen-khich)

---

## 1. Tổng quan Dự án & Ba Câu hỏi Nghiên cứu (`RQ1, RQ2, RQ3`)

Trên bàn cờ `3×3` cổ điển ($3^9 = 19,683$ cấu hình), các thuật toán tìm kiếm toàn cây hoặc bộ luật heuristic chuẩn dễ kéo ván đấu về thế hòa, làm mờ ranh giới khác biệt giữa tìm kiếm cây, mạng học sâu và học tăng cường. Dự án xây dựng bộ Engine **Extended Tic-Tac-Toe (`M×N, K`)** tổng quát (trò chơi đối kháng 2 người, quan sát đầy đủ, tất định, tổng bằng không `zero-sum`) và chọn **`4×4, K=4`** ($3^{16} \approx 4.3 \times 10^7$ cấu hình) làm **chuẩn thực nghiệm chính**, cùng các cấu hình đối chứng **`4×4, K=3`**, **`3×3, K=3`**, và **`5×5, K=4`**.

Thay vì chỉ xếp hạng thắng/thua đơn thuần, hệ thống thu thập **16,800 ván giải đấu chính thức** (4 giải `Paired Round-Robin`, cân bằng 50% `X` / 50% `O`) và **8,756 ván thực nghiệm Ablation** (cùng **84 ván** smoke kiểm chứng `T4.1`) để trả lời có căn cứ thống kê **3 câu hỏi nghiên cứu (RQ1–RQ3)**:

| Mã | Câu hỏi Nghiên cứu Trọng tâm | Kết luận Thực nghiệm Tóm tắt (từ raw `logs/*.jsonl` & [summary_tables.md](./reports/summary_tables.md)) |
|---|---|---|
| **RQ1** | **Hybrid MCTS (AlphaGo-inspired) so với Pure MCTS (UCT) & Policy-only** | Tại ngân sách mô phỏng thấp (`50 sims` trên `4×4, K=4`), `HybridRollout_sim50` (`0.583 [0.542, 0.625]`, chỉ `5` thua) và `HybridMixed_sim50` (`0.575 [0.533, 0.621]`, độ sâu `6.46`) vượt trội cả `PureMCTS_sim50` (`0.529 [0.471, 0.588]`, `22` thua, độ sâu `2.42`) lẫn `Policy_Strong` không tìm kiếm (`0.525`). Ở giới hạn thời gian CPU chặt (`10 ms` / `50 ms`), chi phí forward PyTorch làm giảm thông lượng mô phỏng (`145.9` vs `190.7 sims` ở `10 ms`), nhưng bộ đệm **LRU Cache theo hash bàn cờ** đạt tỷ lệ trúng **`60.7%–69.0%`**. Trên `3×3, K=3` (`default`), `L7_HybridMCTS` dẫn đầu toàn giải (`0.637 [0.624, 0.650]`, `328-872-0`, `0` thua). |
| **RQ2** | **Lợi thế người đi trước ($\Delta\text{Score} = \text{Score}_X - \text{Score}_O$) theo luật thắng `K` & cấp AI** | Khi nới bàn cờ thành `4×4` nhưng giữ `K=3`, lợi thế đi trước bùng nổ với `X Win = 76.5%`, `Draw = 0.0%`, $\Delta\text{Score} = +0.530\;[+0.445, +0.610]$ (`RuleBased`, `AlphaBeta`, `DQN` đạt `1.000` khi đi `X`). Tăng chuỗi thắng lên `4×4, K=4` mở rộng không gian phòng thủ, đẩy tỷ lệ hòa lên `74.1%` (`71.1%–73.3%` trong các giải `4,200` ván) và kéo $\Delta\text{Score}$ về gần cân bằng $+0.024\;[-0.011, +0.058]$ (so với $+0.112\;[+0.073, +0.152]$ trên `3×3, K=3`). |
| **RQ3** | **Độ ổn định đa seed & Tổng quát hóa: RL Self-Play vs Supervised Learning** | Khi gặp các đối thủ cây tìm kiếm chưa từng xuất hiện lúc huấn luyện (*Unseen Opponents*: `MCTS`, `AlphaBeta`), tác tử RL (`DQNAgent` với Mixed Pool 70% Self-Play / 30% Baseline trên `4×4, K=4`, `3 seeds {42, 123, 999}`) đạt **`0.586 ± 0.027`** tổng thể (`2–3` thua / 360 ván) và **`0.554 ± 0.016`** trước Unseen `MCTS`. Ngược lại, `SL_StrongPolicy` giảm xuống **`0.497`** (`30` thua tổng thể, `0.344` trước `MCTS`) và `SL_ValueGreedy` tụt xuống **`0.307`** (`0.256` trước `MCTS`). Trên `3×3, K=3`, `TabularQ` 3 seed đạt **`0.659 ± 0.005`** (`0.619 ± 0.023` trước `MCTS`, `0–1` thua / 360 ván) so với `0.469–0.485` của nhóm Supervised. |

---

## 2. Kiến trúc Hệ thống & 7 Cấp độ Tác tử AI

Hệ thống áp dụng kiến trúc phân lớp lỏng (*loosely coupled architecture*) chia thành 5 phân hệ độc lập: `engine`, `agents`, `training`, `benchmark`, và `ui` (chi tiết sơ đồ và giao ước xem tại [Planning/ARCHITECTURE.md](./Planning/ARCHITECTURE.md)).

- **Giao diện Game API thống nhất (`ExtendedTicTacToe`):** Mọi tác tử AI chỉ tương tác với trò chơi thông qua các phương thức công khai: `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result(player)`, `render()`, `clone()`, và `undo()`, tách biệt khỏi biến nội bộ `_board`/`_history` (được kiểm tra tự động bằng AST tests trong `tests/`).
- **Chống rò rỉ dữ liệu & xác minh checkpoint:** Dữ liệu Supervised chia `train / val / test = 70% / 15% / 15%` theo **`game_id`**; mọi checkpoint `.pt`/`.pkl` đi kèm file `*.meta.json` lưu hash SHA-256 và `board_config` để kiểm chứng hợp lệ trước khi suy luận.

### Hệ thống 7 Cấp độ Tác tử AI

| Cấp | Lớp Tác tử (`src/ttt_ai/agents/`) | Phương pháp Kỹ thuật | Chi phí & Telemetry (`AgentDecision`) |
|---|---|---|---|
| **1** | `RandomAgent` | Baseline chọn đều ngẫu nhiên từ `legal_actions()` qua `np.random.Generator` gắn `seed`. | `nodes=1`, phân phối đều |
| **2** | `RuleBasedAgent` | Tổng quát hóa bộ luật **Newell & Simon** 6 tầng ưu tiên cho mọi `M×N, K`: Thắng ngay $\rightarrow$ Chặn thắng ngay $\rightarrow$ Tạo fork ($\ge 2$ đường mở) $\rightarrow$ Chặn fork $\rightarrow$ `positional_weights()` $\rightarrow$ Ngẫu nhiên có kiểm soát. | Tầng luật & `nodes_visited` |
| **3** | `MinimaxAgent` / `AlphaBetaAgent` | Tìm kiếm đối kháng `step()/undo()` trên 1 bản `clone()`, hàm lượng giá `evaluate_board` theo cửa sổ mở (`tanh`), **Alpha-Beta Pruning**, **Move Ordering**, và **Iterative Deepening** theo `time_budget_ms` (kiểm tra `perf_counter()` mỗi 256 nút). | `nodes_visited`, `depth_reached`, giá trị gốc |
| **4** | `MCTSAgent` | Cây tìm kiếm Monte Carlo **4 pha UCT** (Selection, Expansion, Uniform Random Rollout, Backpropagation); dừng theo `n_simulations` hoặc `time_budget_ms`; giải phóng cây sau mỗi nước đi. | `simulations`, `nodes_visited`, phân phối visit $N(s,a)/N(s)$ |
| **5** | `PolicyNetworkAgent` / `ValueGreedyAgent` | Mạng CNN+MLP `(B, 1, M, N)` gồm **`FastPolicyNet`** (`128` conv + 2 MLP `64`), **`StrongPolicyNet`** (`128` conv + 3 MLP `64`), và **`ValueNet`** (`Tanh` đầu ra $v \in [-1, 1]$). Áp dụng **Action Masking (`-1e9`)** trên ô bất hợp lệ trước `Softmax`/`argmax`. | Xác suất `Softmax` hoặc `-V(s')` |
| **6** | `TabularQAgent` / `DQNAgent` | **`TabularQAgent`** (canonical byte keys cho `3×3 K=3`) và **`DQNAgent`** (`Double DQN` + Experience Replay `50,000` + Target Network + Mixed Pool 70% Self-Play / 30% Baseline trên `4×4`), huấn luyện đa seed `42, 123, 999`. | Giá trị `Q(s, a)` dự đoán |
| **7** | `HybridMCTSAgent` *(AlphaGo-inspired)* | Kết hợp cây tìm kiếm **PUCT** với prior $P(s, a)$ từ `StrongPolicyNet`, **LRU Cache theo canonical hash**, và 2 chế độ đánh giá nút lá (`rollout` hoặc `mixed` $\lambda V_\theta(s) + (1-\lambda)z_{\text{rollout}}$). Phát đồng thời cả `prior_probs` lẫn `action_scores`. | Đồng thời `Prior P(s,a)` & `Visit Share N(s,a)/N(s)` |

> **Minh bạch phạm vi Cấp 7:** `HybridMCTSAgent` được định vị là mô hình **AlphaGo-inspired** (Supervised `PolicyNet` + `ValueNet` kết hợp cây PUCT trên CPU), không tái hiện kiến trúc phân tán đa GPU/TPU hay vòng lặp tự học ResNet hoàn chỉnh của AlphaZero nguyên bản.

---

## 3. Bộ Tài liệu Sổ tay (`Planning/`) & Cấu trúc Thư mục Dự án

### Lộ trình đọc bộ tài liệu `Planning/` (`01 → 06`)
Bộ 6 tài liệu trong `Planning/` được biên soạn theo cấu trúc sổ tay kỹ thuật để tự định hướng và rà soát xuyên suốt quá trình phát triển và thực nghiệm. Để nắm mạch thiết kế từ ý tưởng đến kết quả, có thể đọc theo thứ tự:

| Thứ tự | Tài liệu | Nội dung chính |
|---|---|---|
| **01** | [PRD.md](./Planning/PRD.md) | Mục tiêu đề tài, 3 câu hỏi nghiên cứu (`RQ1–RQ3`), mô hình MDP `M×N, K`, 7 cấp AI & Rubric |
| **02** | [ARCHITECTURE.md](./Planning/ARCHITECTURE.md) | Kiến trúc phân lớp 5 phân hệ, sơ đồ luồng, Game API, chi tiết thuật toán & cấu trúc repo |
| **03** | [DESIGN.md](./Planning/DESIGN.md) | Thiết kế `Rich CLI` Dashboard, thang nhiệt và quy chuẩn 5 biểu đồ (`Fig 1..5`) |
| **04** | [RULES.md](./Planning/RULES.md) | Nguyên tắc trung thực học thuật cốt lõi, quy ước viết mã PEP 8 và chuẩn đo lường |
| **05** | [TASKS.md](./Planning/TASKS.md) | Kế hoạch phân rã 5 Sprint (`T1.1`–`T5.3`) và tiêu chí tự kiểm chứng từng giai đoạn |
| **06** | [PROGRESS.md](./Planning/PROGRESS.md) | Nhật ký thực nghiệm (`25,640` ván logged), bảng số liệu chính thức & lịch sử gỡ lỗi (`BUG-01..04`) |

### Cấu trúc Thư mục Dự án

```text
Extended_Tic-Tac-Toe_Project/
├── Planning/                         # Bộ 6 sổ tay thiết kế, quy chuẩn & nhật ký (01 -> 06)
│   ├── PRD.md                        # 01. Định hướng đề tài, 3 RQ (RQ1–RQ3), MDP & Rubric
│   ├── ARCHITECTURE.md               # 02. Sổ tay kiến trúc hệ thống, schemas & thuật toán
│   ├── DESIGN.md                     # 03. Thiết kế giao diện Rich CLI & Fig 1..5
│   ├── RULES.md                      # 04. Nguyên tắc trung thực học thuật & quy ước kỹ thuật
│   ├── TASKS.md                      # 05. Kế hoạch triển khai 5 Sprint (T1.1–T5.3)
│   └── PROGRESS.md                   # 06. Nhật ký thực nghiệm, Bug Log (BUG-01..04) & Registry
├── Reports_Project/                  # Bộ sản phẩm báo cáo học thuật, thuyết trình & poster
│   ├── LaTeX/                        # Mã nguồn báo cáo đồ án LaTeX hoàn chỉnh theo chuẩn SGU
│   │   ├── main.tex                  # Báo cáo toàn văn 7 chương (74 KB, công thức toán, bảng & phân tích)
│   │   ├── SGU-LOGO.png              # Huy hiệu / Logo chính thức Trường Đại học Sài Gòn
│   │   └── reports/figures/          # 5 biểu đồ thực nghiệm độ phân giải cao 300 DPI (fig1..5)
│   ├── Slide_Extended_TicTacToe.pptx # Slide báo cáo thuyết trình đồ án cuối kỳ (13 slide chuyên nghiệp)
│   └── Poster_A3_Extended_TicTacToe.pdf # Poster báo cáo khoa học khổ A3 chuẩn hội nghị học thuật
├── Jupiter_Notebook/                 # Sổ tay thực nghiệm tương tác Jupyter Notebook
│   └── Extended_Tic_Tac_Toe.ipynb    # Notebook 6 phần: Game API, 7 AI, Heatmap, tái tạo Fig 1..5 & RQ1-3
├── References/                       # Đề cương, slide, sách & kho mã tham khảo (chỉ đọc)
│   ├── Tic tac toe Game.pdf
│   ├── 2024 - Liu - AlphaGo Simplified Rule-Based AI and Deep Learning in Everyday Games.pdf
│   ├── Artificial Intelligence A Modern Approach 4th.pdf
│   ├── SuttonBartoIPRLBook2ndEd.pdf
│   ├── AlphaGoSimplified-main/       # Mã nguồn tham khảo minh họa từ sách Mark Liu (2024)
│   └── Reversi-mcts-master/          # Mã nguồn tham khảo cấu trúc MCTS UCT từ Reversi C#
├── configs/                          # 7 tệp cấu hình YAML tập trung
│   ├── game_3x3_k3.yaml              # Cấu hình môi trường đối chứng 3x3 K=3
│   ├── game_4x4_k3.yaml              # Cấu hình môi trường bất đối xứng 4x4 K=3 (RQ2)
│   ├── game_4x4_k4.yaml              # Cấu hình môi trường chuẩn thực nghiệm chính 4x4 K=4
│   ├── game_5x5_k4.yaml              # Cấu hình môi trường mở rộng 5x5 K=4 (Bonus)
│   ├── train_supervised.yaml         # Siêu tham số huấn luyện SL Cấp 5
│   ├── train_rl.yaml                 # Siêu tham số huấn luyện Tabular Q & DQN Self-Play Cấp 6
│   └── tournament.yaml               # Thiết lập giải đấu Paired Round-Robin & ngân sách tính toán
├── src/ttt_ai/                       # Gói thư viện lõi (Python package, cài đặt dạng editable)
│   ├── config.py                     # Typed Dataclasses, YAML loader & seed_everything()
│   ├── engine/                       # exceptions.py, rules.py, game.py (ExtendedTicTacToe core)
│   ├── agents/                       # base.py + 7 cấp AI (nn_models.py, hybrid_mcts_agent.py, ...)
│   ├── training/                     # dataset_gen.py, train_supervised.py, train_q_tabular.py, train_dqn_selfplay.py, checkpoint_io.py
│   ├── benchmark/                    # tournament.py, metrics.py, ablations.py, visualizer.py
│   └── ui/                           # theme.py, cli_app.py (InteractiveCLIApp + Setup Wizard + Heatmap)
├── scripts/                          # 6 CLI entrypoints tái lập thực nghiệm & demo
│   ├── generate_dataset.py           # Sinh dữ liệu chuyên gia & huấn luyện mạng SL Cấp 5
│   ├── train_all.py                  # Huấn luyện toàn diện Cấp 6 (Tabular Q & DQN đa seed)
│   ├── run_tournament.py             # Điều phối 4 giải đấu Paired Round-Robin (16,800 ván)
│   ├── run_ablations.py              # Điều phối 4 kịch bản Ablation khoa học (8,756 ván)
│   ├── plot_results.py               # Kết xuất tự động Fig 1..5 (300 DPI) & bảng tổng hợp MD/CSV
│   └── play.py                       # Khởi chạy Rich CLI Dashboard (Human vs AI & AI vs AI)
├── tests/                            # conftest.py + 18 module kiểm thử tự động (347 test cases pass 100%)
├── checkpoints/                      # 15 cặp trọng số đã huấn luyện (.pt/.pkl) + *.meta.json kèm SHA-256
├── logs/                             # Datasets (.npz), logs giải đấu (16,800 ván), ablations (8,756 ván), train logs
├── reports/                          # figures/fig1..5*.png (300 DPI) + CSV/MD tổng hợp & sample_failure_cases.md
├── pyproject.toml                    # Cấu hình đóng gói dự án, phụ thuộc pip & thiết lập Ruff / Pytest
└── README.md                         # Tài liệu thuyết minh tổng quan dự án
```

---

## 4. Bộ Sản phẩm Báo cáo, Thuyết trình & Sổ tay Thực nghiệm (`Reports_Project/` & `Jupiter_Notebook/`)

Bên cạnh mã nguồn và bộ tài liệu kỹ thuật `Planning/`, dự án cung cấp đầy đủ bộ **4 ấn phẩm học thuật chính thức** phục vụ việc báo cáo đồ án và tái lập kết quả tương tác:

| Sản phẩm | Vị trí Tệp / Thư mục | Định dạng & Quy mô | Mục đích & Đối tượng sử dụng |
|---|---|---|---|
| **Báo cáo đồ án** | [`Reports_Project/LaTeX/`](./Reports_Project/LaTeX/) | LaTeX source (`main.tex`, ~74 KB, 7 chương) | Báo cáo đồ án cuối kỳ hoàn chỉnh theo chuẩn học thuật SGU, kèm công thức toán học, bảng biểu và trích dẫn hình ảnh 300 DPI. |
| **Slide Thuyết trình Đồ án** | [`Reports_Project/Slide_Extended_TicTacToe.pptx`](./Reports_Project/Slide_Extended_TicTacToe.pptx) | PowerPoint presentation (13 slide, 16:9, ~2.8 MB) | Trình bày trước Giảng viên hướng dẫn; cô đọng toàn bộ lộ trình, điểm nhấn thuật toán, kết quả 16,800 ván và demo. |
| **Poster Khoa học Khổ A3** | [`Reports_Project/Poster_A3_Extended_TicTacToe.pdf`](./Reports_Project/Poster_A3_Extended_TicTacToe.pdf) | PDF Vector (1 trang A3 đứng, ~3.2 MB) | Trưng bày báo cáo poster; tổng hợp cô đọng toàn bộ đồ án chỉ trên một giao diện trực quan duy nhất. |
| **Sổ tay Thực nghiệm Tương tác** | [`Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb`](./Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb) | Jupyter Notebook (22 cells, 6 phần) | Môi trường tương tác trực quan (Interactive Research Lab) cho phép chạy thử Game Engine, đấu thử 7 AI, hiển thị Heatmap và tái hiện trực tiếp Fig 1..5. |

---

### 4.1. Báo cáo đồ án bằng LaTeX (`Reports_Project/LaTeX/`)

Thư mục [`Reports_Project/LaTeX/`](./Reports_Project/LaTeX/) chứa toàn bộ mã nguồn của báo cáo đồ án chính thức được soạn thảo bằng **LaTeX** theo quy chuẩn học thuật khắt khe của Trường Đại học Sài Gòn (SGU).

- **Tệp nguồn chính:** [`Reports_Project/LaTeX/main.tex`](./Reports_Project/LaTeX/main.tex) (dung lượng hơn 74 KB, biên dịch qua `pdflatex` hoặc `xelatex`).
- **Nhận diện thương hiệu:** Gắn kèm huy hiệu chính thức của nhà trường [`SGU-LOGO.png`](./Reports_Project/LaTeX/SGU-LOGO.png) tại trang bìa trang trọng.
- **Thư mục hình ảnh:** [`Reports_Project/LaTeX/reports/figures/`](./Reports_Project/LaTeX/reports/figures/) chứa đầy đủ 5 biểu đồ thực nghiệm chuẩn 300 DPI (`fig1` đến `fig5`), được liên kết trực tiếp vào văn bản thông qua môi trường `\begin{figure}[H]`.

#### Cấu trúc 7 Chương Nội dung của Báo cáo:
1. **Chương 1: Giới thiệu & Đặt vấn đề Nghiên cứu:** Phân tích bước nhảy tổ hợp từ $3 \times 3$ ($3^9$) lên $4 \times 4, K=4$ ($3^{16} \approx 4.3 \times 10^7$ trạng thái), phát biểu mục tiêu kỹ thuật và 3 câu hỏi nghiên cứu trọng tâm (**RQ1–RQ3**).
2. **Chương 2: Mô hình hóa Toán học & Kiến trúc Hệ thống:** Hình thức hóa bài toán dưới dạng Markov Decision Process (MDP) tổng bằng không; trình bày chi tiết kiến trúc phân tầng 5 module độc lập (`engine`, `agents`, `training`, `benchmark`, `ui`) và hợp đồng Game API nghiêm ngặt.
3. **Chương 3: Hiện thực hóa 7 Cấp độ Tác tử AI:** Trình bày chi tiết lý thuyết, công thức toán học và thiết kế thuật toán từ Cấp 1 (`RandomAgent`), Cấp 2 (`RuleBasedAgent` 6 tầng Newell & Simon), Cấp 3 (`MinimaxAgent` & `AlphaBetaAgent` kèm Iterative Deepening), Cấp 4 (`MCTSAgent` 4 pha UCT), Cấp 5 (`Fast/StrongPolicyNet` và `ValueNet` có Action Masking), Cấp 6 (`TabularQAgent` và `DQNAgent` Self-Play + Mixed Pool), đến Cấp 7 (`HybridMCTSAgent` PUCT + Policy Prior + LRU Cache).
4. **Chương 4: Thiết kế Đánh giá, Giải đấu & Kết quả Thực nghiệm:** Mô tả giao thức giải đấu Paired Round-Robin (16,800 ván đấu chính thức, cân bằng tuyệt đối 50% X / 50% O), đo đạc trên cả chế độ *Default* lẫn *Fixed Compute Budget (10 ms, 50 ms)*, 4 bộ thực nghiệm Ablation (8,756 ván), ước lượng khoảng tin cậy 95% (Wilson Score và Bootstrap), và đưa ra kết luận định lượng rõ ràng cho từng câu hỏi **RQ1, RQ2, RQ3**.
5. **Chương 5: Giao diện Rich CLI, Phân tích Lỗi & Thảo luận:** Thiết kế bảng điều khiển Live Match Dashboard (48% bàn cờ / 52% telemetry & heatmap 5 mức màu), minh họa cơ chế Explainability thời gian thực đối chiếu `Prior P(s,a)` vs `Visit Share N(s,a)/N(s)`, và truy vết raw logs giải mã 4 ca thất bại chiến thuật điển hình (*Fork trap*, *Horizon effect*, *Overconfidence*, *Simulation starvation*).
6. **Chương 6: Kết luận & Đối chiếu Rubric Đồ án:** Đánh giá tổng kết các đóng góp cốt lõi của công trình, bảng tự đánh giá đối chiếu chi tiết 100 điểm Rubric môn học + 10 điểm thưởng khuyến khích, cùng thảo luận về các giới hạn và hướng mở rộng tương lai.
7. **Tài liệu Tham khảo:** Trích dẫn các công trình kinh điển của Russell & Norvig (*AIMA 4th*), Sutton & Barto (*RL 2nd*), Silver et al. (*AlphaGo / AlphaZero*), và Mark Liu (2024).

#### Hướng dẫn Biên dịch Báo cáo LaTeX:
```bash
# Di chuyển vào thư mục LaTeX
cd Reports_Project/LaTeX

# Biên dịch bằng pdflatex (chạy 2 lần để cập nhật mục lục và liên kết chéo)
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
*(Hoặc nén toàn bộ thư mục `Reports_Project/LaTeX/` thành file `.zip` và tải lên Overleaf để biên dịch trực tuyến với TeX Live 2024+).*

---

### 4.2. Slide Báo cáo Thuyết trình Đồ án (`Reports_Project/Slide_Extended_TicTacToe.pptx`)

Tệp trình chiếu [`Reports_Project/Slide_Extended_TicTacToe.pptx`](./Reports_Project/Slide_Extended_TicTacToe.pptx) được thiết kế chuyên nghiệp theo tỷ lệ chuẩn **16:9**, sử dụng bảng màu nhận diện thương hiệu học thuật đồng bộ với Trường ĐH Sài Gòn, tối ưu hóa cho bài thuyết trình bảo vệ đồ án trước Hội đồng (thời lượng 15–20 phút).

#### Phân bổ Chi tiết 13 Slide Thuyết trình:
- **Slide 01 — Tiêu đề & Thông tin Đồ án:** Nhận diện SGU, Khoa CNTT, Tên học phần, Đề tài *"Xây dựng và đánh giá Game AI đa cấp độ — Extended Tic-Tac-Toe"*, Giảng viên hướng dẫn ThS. Đỗ Như Tài và Sinh viên thực hiện Trương Chí Long.
- **Slide 02 — 01. Đặt vấn đề & Động lực Nghiên cứu:** Bước nhảy không gian trạng thái từ $3 \times 3$ lên $4 \times 4, K=4$ phá vỡ thế cân bằng hòa tầm thường và bộc lộ sự phân hóa năng lực giữa các phương pháp AI.
- **Slide 03 — 02. Ba Câu hỏi Nghiên cứu Trọng tâm:** Trình bày rõ ràng 3 câu hỏi định lượng cốt lõi **RQ1** (Hybrid MCTS vs Pure MCTS/Policy-only), **RQ2** (Lợi thế đi trước theo luật thắng $K$), và **RQ3** (RL Self-Play vs Supervised Learning trước Unseen Opponents).
- **Slide 04 — 03. Kiến trúc Hệ thống Phân tầng (`ttt_ai`):** Sơ đồ 5 phân hệ độc lập, nguyên tắc cô lập biến nội bộ lõi Engine, hợp đồng Game API chuẩn và cơ chế chia tập dữ liệu theo `game_id` chống rò rỉ dữ liệu.
- **Slide 05 — 04. Hệ sinh thái 7 Cấp độ Tác tử AI:** Bảng tổng hợp tiến hóa từ Baseline ngẫu nhiên (L1), Hệ luật Heuristic (L2), Tìm kiếm cây đối kháng (L3, L4), Mạng học sâu (L5), Học tăng cường (L6) đến Mô hình lai PUCT cảm hứng AlphaGo (L7).
- **Slide 06 — 05. Điểm nhấn Thuật toán Nâng cao (Cấp 3, 6, 7):** Cắt giảm **-94.71% nút duyệt** ở $d=4$ với Alpha-Beta Pruning; cơ chế Double DQN Self-Play kết hợp Mixed Opponent Pool; thuật toán PUCT lai Policy/Value với LRU Hash Cache đạt tỷ lệ trúng 60.7%–69.0%.
- **Slide 07 — 06. Thiết kế Thực nghiệm & Giải đấu Công bằng:** Thể thức Paired Round-Robin cân bằng 50/50 X/O (200 ván/cặp), đối chiếu giữa Default vs Fixed Budget (10 ms, 50 ms), và phương pháp ước lượng khoảng tin cậy 95% Wilson / Bootstrap.
- **Slide 08 — 07. Kết quả Giải đấu Chính thức (16,800 ván):** Bảng xếp hạng điểm số tổng thể, phân tích tỷ lệ hòa thực tế đạt 71.1%–74.1% trên bàn $4 \times 4, K=4$ chứng minh tính mở và giàu tính chiến thuật của môi trường.
- **Slide 09 — 08. Thí nghiệm Trả lời RQ1 (Ablation 2 — 2,760 ván):** Minh chứng định lượng sự vượt trội của Hybrid MCTS ở ngân sách mô phỏng thấp (`50 sims`: Score 0.583 vs 0.529 của Pure MCTS và 0.525 của Strong Policy).
- **Slide 10 — 09. Thí nghiệm Trả lời RQ2 & RQ3:** Bùng nổ lợi thế đi trước ở $4 \times 4, K=3$ ($\Delta\text{Score} = +0.530$) và sự sụp đổ về thế cân bằng ở $K=4$ ($\Delta\text{Score} = +0.024$); Khả năng tổng quát hóa vượt trội của DQN Self-Play (`0.586 ± 0.027`) so với Supervised (`0.497`) trước đối thủ cây tìm kiếm chưa từng gặp.
- **Slide 11 — 10. Giao diện Rich CLI & Giải thích Thời gian thực:** Bố cục Live Match Dashboard (48% bàn cờ / 52% telemetry), Action Heatmap 5 mức màu, và minh họa trực quan sự hiệu chỉnh từ `Prior P(s,a)` sang `Visit Share N(s,a)/N(s)` ở Cấp 7.
- **Slide 12 — 11. Phân tích Ca lỗi & Bài học Kỹ thuật:** Truy vết từ raw logs 4 dạng thất bại chiến thuật điển hình (*Bẫy Fork đôi*, *Hiệu ứng chân trời*, *Ảo tưởng tự tin mạng nơ-ron*, *Nghèo đói mô phỏng MCTS*).
- **Slide 13 — 12. Tổng kết & Ánh xạ Rubric Đồ án:** Hoàn thành trọn vẹn 5/5 Sprint, 347/347 test cases pass 100%, đối chiếu đạt 100/100 điểm chuẩn Rubric + 10/10 điểm khuyến khích.

---

### 4.3. Poster Báo cáo Khoa học Khổ A3 (`Reports_Project/Poster_A3_Extended_TicTacToe.pdf`)

Tệp [`Reports_Project/Poster_A3_Extended_TicTacToe.pdf`](./Reports_Project/Poster_A3_Extended_TicTacToe.pdf) là poster báo cáo khoa học khổ **A3** (dạng trang ngang, kích thước $297 \times 420\text{ mm}$, định dạng PDF vector chất lượng in ấn cao), được trình bày theo chuẩn mực hội nghị khoa học quốc tế.

#### Các Khối Nội dung Cốt lõi trên Poster:
1. **Header & Metadata:** Huy hiệu Trường ĐH Sài Gòn, Khoa CNTT, Học phần TTNTNC (`841458`), Tiêu đề song ngữ Việt - Anh, Tác giả Trương Chí Long (MSSV: `3123410198`), GVHD ThS. Đỗ Như Tài và đường dẫn GitHub repository.
2. **Khối Thông số Nổi bật (Key Metrics Banner):**
   - **$4 \times 4, K=4$ MDP:** Môi trường chuẩn $3^{16} \approx 4.3 \times 10^7$ cấu hình; mở rộng linh hoạt từ $3 \times 3$ đến $5 \times 5$.
   - **7 Cấp độ Tác tử (L1–L7):** Heuristic, Alpha-Beta, Pure MCTS, Deep Policy/Value, Double DQN Self-Play & Hybrid PUCT.
   - **25,640 Ván · 347 Tests:** 16,800 ván giải đấu + 8,756 ván ablation + 84 ván smoke; 347/347 pytest pass 100% trong 157 giây; Ruff PEP 8 clean tuyệt đối.
3. **Panel 1 — Đặt vấn đề, Bước nhảy Tổ hợp & RQ1–RQ3:** Phân tích sự sụp đổ của bài toán $3 \times 3$ và thiết kế 3 câu hỏi định lượng cốt lõi.
4. **Panel 2 — Kiến trúc Phân tầng 5 Module & API Chuẩn:** Sơ đồ 5 module (`ui`, `benchmark`, `agents`, `training`, `engine`) và hợp đồng Game API cô lập trạng thái.
5. **Panel 3 — Hệ sinh thái 7 Cấp độ Tác tử AI:** Bảng phân tầng thuật toán, cơ chế Action Masking (`-1e9`), Move Ordering, và PUCT Hybrid Search.
6. **Panel 4 — Phương pháp Đánh giá Thực nghiệm:** Giao thức Paired Round-Robin cân bằng 50/50 X/O, kiểm soát ngân sách tính toán (Default, 10 ms, 50 ms) và kiểm định khoảng tin cậy 95%.
7. **Panel 5 — Trực quan hóa Kết quả Thực nghiệm (5 Biểu đồ Fig 1–5):** Thu nhỏ sắc nét 5 biểu đồ khoa học: Ma trận đối đầu kép, độ trễ/chi phí tìm kiếm, kết quả RQ1, RQ2 và RQ3.
8. **Panel 6 — Explainability & Rich CLI Live Dashboard:** Bố cục Dashboard terminal tương tác, Action Heatmap 5 mức màu và cơ chế giải thích nước đi Cấp 7.
9. **Panel 7 — Kết luận & Đóng góp:** Tóm tắt 4 đóng góp khoa học chính và xác nhận chuẩn mực trung thực học thuật (*Zero Fabrication*).

---

### 4.4. Sổ tay Thực nghiệm Tương tác Jupyter Notebook (`Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb`)

Tệp [`Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb`](./Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb) là sổ tay nghiên cứu tương tác gồm **22 cells** (kết hợp giải thích Markdown khoa học và mã nguồn Python có thể thực thi độc lập). Đây là một công cụ lý tưởng cho việc thẩm định mã nguồn, giảng dạy trực quan hoặc kiểm chứng nhanh từng thành phần hệ thống mà không cần mở Terminal.

#### Lộ trình 6 Phần Thực nghiệm trong Notebook:
- **Phần 1: Tổng quan Đồ án & Thiết lập Môi trường (Cells 0–1):**
  - Giới thiệu bài toán, thiết lập đường dẫn `src/` vào `sys.path`, khởi tạo thư viện khoa học (`numpy`, `torch`, `matplotlib`, `yaml`), và cố định hạt giống ngẫu nhiên toàn cục `seed_everything(42)`.
- **Phần 2: Lõi Trò chơi & Giao diện Hợp đồng Engine (Cells 2–4):**
  - Nạp cấu hình từ `configs/game_4x4_k4.yaml` và `configs/game_3x3_k3.yaml`.
  - Khởi tạo môi trường `ExtendedTicTacToe`, thực nghiệm các phương thức chuẩn: `reset()`, `step()`, `undo()`, `clone()`, `check_win_around()` với độ phức tạp tối ưu $O(K)$, tính toán góc nhìn đối xứng `canonical_board()`, và hiển thị bàn cờ dạng chuỗi.
- **Phần 3: Thi triển & Đối đầu 7 Cấp độ Tác tử AI (Cells 5–9):**
  - Khởi tạo trực tiếp từng tác tử: Cấp 1 (`RandomAgent`), Cấp 2 (`RuleBasedAgent`), Cấp 3 (`AlphaBetaAgent`), Cấp 4 (`MCTSAgent`).
  - Xác minh tính toàn vẹn của trọng số mạng nơ-ron thông qua tệp `*.meta.json` và mã băm SHA-256 (`checkpoint_io.py`), nạp các tác tử Cấp 5 (`PolicyNetworkAgent`, `ValueGreedyAgent`) và Cấp 6 (`TabularQAgent`, `DQNAgent`).
  - Trình diễn Cấp 7 (`HybridMCTSAgent` PUCT lai mạng nơ-ron), thu thập và in ra toàn bộ trường vi mô trong đối tượng `AgentDecision` (`latency_ms`, `nodes_visited`, `depth_reached`, `prior_probs`, `action_scores`).
- **Phần 4: Action Heatmap & Trực quan hóa Thời gian thực (Cells 10–11):**
  - Cài đặt hàm vẽ bản đồ nhiệt Matplotlib 5 mức màu tương thích trực tiếp với quy chuẩn đồ họa trong [Planning/DESIGN.md](./Planning/DESIGN.md).
  - Trực quan hóa một thế cờ trung cuộc gay cấn trên bàn $4 \times 4, K=4$; đặt cạnh nhau để so sánh trực tiếp giữa gợi ý trực giác ban đầu `Prior P(s,a)` của mạng Policy và quyết định điều chỉnh sau duyệt cây `Visit Share N(s,a)/N(s)` của Cấp 7 (minh chứng trực quan cho **RQ1**).
- **Phần 5: Tái tạo Kết quả Thực nghiệm Đồ án (`Fig 1–5` & `RQ1, RQ2, RQ3`) (Cells 12–20):**
  - Đọc trực tiếp dữ liệu từ các tệp log thô `logs/tournament_*.jsonl`, `logs/ablation_*.jsonl` và báo cáo tổng hợp `reports/summary_metrics.csv`.
  - Tái tạo đồ họa độ phân giải cao cho toàn bộ 5 hình khoa học:
    - **Fig 1:** Ma trận đối đầu kép 7 tác tử (Tỷ lệ thắng và điểm số khi cầm quân X vs khi cầm quân O).
    - **Fig 2:** Độ trễ suy nghĩ `mean_ms` (thang log), phân vị 95 `p95_ms` và số nút/mô phỏng trung bình.
    - **Fig 3 (RQ1):** Đường cong năng lực của Hybrid MCTS (`rollout` & `mixed`) so với Pure MCTS và Policy-only theo ngân sách simulations $\in \{50, 150, 400\}$ và ngân sách thời gian 10 ms / 50 ms.
    - **Fig 4 (RQ2):** Lợi thế đi trước $\Delta\text{Score}$ kèm khoảng tin cậy 95% trên các cấu hình $4 \times 4, K=3$, $3 \times 3, K=3$, và $4 \times 4, K=4$.
    - **Fig 5 (RQ3):** Đường cong hội tụ huấn luyện và khả năng tổng quát hóa của DQN Self-Play (3 seeds) trước các đối thủ cây tìm kiếm chưa từng gặp lúc huấn luyện.
  - Tích hợp các đoạn văn bản thảo luận và phân tích định lượng chuyên sâu cho từng câu hỏi nghiên cứu ngay bên dưới mỗi biểu đồ.
- **Phần 6: Tổng kết & Kết luận (Cells 21):**
  - Trình bày bảng đối chiếu Rubric môn học (100/100 điểm chuẩn + 10 điểm khuyến khích), tóm tắt các đóng góp khoa học, phân tích các điểm hạn chế và vạch ra định hướng mở rộng (AlphaZero hoàn chỉnh, ResNet sâu hơn, mở rộng lên cờ caro $15 \times 15, K=5$).

#### Hướng dẫn Khởi chạy Jupyter Notebook:
```bash
# Kích hoạt môi trường ảo
source .venv/bin/activate  # Trên Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Trên Windows PowerShell

# Khởi chạy Jupyter Lab hoặc Notebook
jupyter lab Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb
# hoặc: jupyter notebook Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb
```
*(Người dùng cũng có thể mở trực tiếp tệp notebook này trong Visual Studio Code với Python Extension đã chọn kernel `.venv`).*

---

## 5. Hướng dẫn Cài đặt & Khởi tạo Môi trường (Quick Start)

Yêu cầu hệ thống: **Python `>= 3.11`** (khuyến nghị `3.11` – `3.12`; đã kiểm thử đầy đủ trên `Python 3.14.7` Windows 11).

Di chuyển vào thư mục gốc dự án `Extended_Tic-Tac-Toe_Project/` và thiết lập môi trường ảo:

### Windows (PowerShell)
```powershell
cd Extended_Tic-Tac-Toe_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows (CMD)
```cmd
cd Extended_Tic-Tac-Toe_Project
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Linux / macOS (Bash/Zsh)
```bash
cd Extended_Tic-Tac-Toe_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

> **Lưu ý:** Lệnh `pip install -e ".[dev]"` tự động cài đặt toàn bộ thư viện lõi (`numpy`, `torch`, `pyyaml`, `rich`, `matplotlib`, `scipy`) và công cụ kiểm thử (`pytest`, `ruff`) từ [pyproject.toml](./pyproject.toml), đồng thời đăng ký gói `ttt_ai` ở chế độ editable. Nếu chỉ cần chạy suy luận/demo mà không cần dev tools, có thể dùng `pip install -e .`.

---

## 6. Hướng dẫn Tái lập Thực nghiệm 1 Lệnh

Mọi bước kiểm thử, sinh dữ liệu, huấn luyện, đấu giải, ablation và vẽ biểu đồ đều có thể chạy lại chỉ bằng **một lệnh duy nhất** từ thư mục `Extended_Tic-Tac-Toe_Project/`:

### 6.1. Kiểm thử Đơn vị, Tích hợp & Kiểm tra Phong cách Code
```bash
# Chạy toàn bộ 347 test cases (engine, 7 agents, AST rules, training, tournament, metrics, ablations, CLI, visualizer)
pytest -v

# Kiểm tra PEP 8, độ dài dòng <= 100 và import sorting với Ruff
ruff check src/ tests/ scripts/
```

### 6.2. Sinh dữ liệu Chuyên gia & Huấn luyện Cấp 5 (Supervised Learning)
Dữ liệu được sinh từ `AlphaBetaAgent` kết hợp `RuleBasedAgent` (có nhiễu mở đầu) và chia `70% / 15% / 15%` theo **`game_id`** để tránh rò rỉ dữ liệu:
```bash
# Sinh toàn bộ dataset (4x4_k4 & 3x3_k3) VÀ huấn luyện đủ FastPolicyNet, StrongPolicyNet, ValueNet
python scripts/generate_dataset.py --board all --train

# Hoặc chạy riêng cho bàn cờ chính 4x4, K=4:
python scripts/generate_dataset.py --board 4x4_k4 --train --seed 42
```

### 6.3. Huấn luyện Cấp 6 Đa Seed (Reinforcement Learning)
Script [scripts/train_all.py](./scripts/train_all.py) điều phối trực tiếp hai mô-đun [train_q_tabular.py](./src/ttt_ai/training/train_q_tabular.py) (`TabularQAgent` trên `3×3 K=3`) và [train_dqn_selfplay.py](./src/ttt_ai/training/train_dqn_selfplay.py) (`DQNAgent` Self-Play + Mixed Opponent Pool trên `4×4 K=4` & `4×4 K=3`) qua bộ 3 seeds `[42, 123, 999]`:
```bash
# Huấn luyện đầy đủ TabularQ (3x3) và DQN Self-Play (4x4_k4, 4x4_k3) qua cả 3 seeds (42, 123, 999)
python scripts/train_all.py --mode all --board all

# Hoặc chạy tách riêng từng nhánh:
python scripts/train_all.py --mode tabular --board 3x3_k3 --seeds 42 123 999
python scripts/train_all.py --mode dqn --board 4x4_k4 --seeds 42 123 999
```

### 6.4. Chạy Giải đấu Paired Round-Robin Chính thức (16,800 ván)
```bash
# Chạy đủ bộ 4 giải đấu chính thức (4x4 default, fixed_10ms, fixed_50ms & 3x3 default; 200 ván/cặp)
python scripts/run_tournament.py --suite

# Kiểm tra tính toàn vẹn & tái tổng hợp bảng báo cáo từ logs có sẵn (không chạy lại trận):
python scripts/run_tournament.py --suite --verify-only

# Hoặc chạy 1 giải đơn lẻ (ví dụ 4x4 K=4, giới hạn 10 ms/nước):
python scripts/run_tournament.py --board 4x4_k4 --budget-mode fixed_10ms --n-pairs 100 --seed 42
```

### 6.5. Chạy 4 Thí nghiệm Ablation Khoa học (`AB Pruning`, `RQ1`, `RQ2`, `RQ3`)
```bash
# Chạy trọn bộ 4 kịch bản ablation ở quy mô đầy đủ (8,756 ván) và xuất báo cáo ablation_*.{csv,md}
python scripts/run_ablations.py --ablation all --mode full --seed 42

# Chế độ chạy thử nhanh (smoke check) cho 1 kịch bản cụ thể (ví dụ RQ1):
python scripts/run_ablations.py --ablation 2 --mode smoke --seed 42
```

### 6.6. Kết xuất Tự động Biểu đồ (`Fig 1..5`, 300 DPI) & Báo cáo Thực nghiệm
Đọc trực tiếp từ raw `logs/*.jsonl` (không chỉnh sửa tay số liệu) để xuất toàn bộ 5 hình PNG `300 DPI` và 3 file báo cáo định lượng (`reports/summary_metrics.csv`, `reports/summary_tables.md`, `reports/sample_failure_cases.md`):
```bash
# Xuất đồng thời cả 5 biểu đồ (300 DPI) và toàn bộ bảng báo cáo / sample failure cases
python scripts/plot_results.py --figures all --dpi 300

# Hoặc chỉ vẽ chọn lọc vài biểu đồ cụ thể (ví dụ Fig 3 và Fig 5):
python scripts/plot_results.py --figures 3 5 --dpi 300
```

| Biểu đồ | Đường dẫn Tệp Đầu ra | Nội dung Minh họa Chính |
|---|---|---|
| **Fig 1** | [`reports/figures/fig1_head_to_head_matrix.png`](./reports/figures/fig1_head_to_head_matrix.png) | Hai ma trận nhiệt đối đầu kép (`As X` vs `As O`) kèm Score và `(W-D-L)` cho cả 7 cấp AI trên `4×4, K=4` |
| **Fig 2** | [`reports/figures/fig2_latency_and_nodes.png`](./reports/figures/fig2_latency_and_nodes.png) | Biểu đồ cột thang log độ trễ `mean_ms` + marker `p95_ms` và chi phí `Avg Nodes` / `Avg Sims` |
| **Fig 3** | [`reports/figures/fig3_ablation_rq1_and_alphabeta.png`](./reports/figures/fig3_ablation_rq1_and_alphabeta.png) | **RQ1:** `Pure MCTS` vs `Hybrid (Rollout/Mixed)` theo `n_simulations ∈ {50, 150, 400}` & `10/50 ms` + đối chứng Alpha-Beta Pruning & Ordering |
| **Fig 4** | [`reports/figures/fig4_ablation_rq2_seat_advantage.png`](./reports/figures/fig4_ablation_rq2_seat_advantage.png) | **RQ2:** Dịch chuyển kết quả (`X Win / Draw / O Win`) và $\Delta\text{Score}$ kèm 95% CI trên `4×4 K=3`, `3×3 K=3`, `4×4 K=4` |
| **Fig 5** | [`reports/figures/fig5_ablation_rq3_learning_and_generalization.png`](./reports/figures/fig5_ablation_rq3_learning_and_generalization.png) | **RQ3:** Learning curves (`Supervised` & `RL 3-seed mean ± std`) và khả năng tổng quát hóa trước đối thủ unseen (`MCTS`, `AlphaBeta`) |

---

## 7. Hướng dẫn Trải nghiệm Trực quan (`Rich CLI` Dashboard)

### Các lệnh khởi chạy `scripts/play.py`

```bash
# 1. Mở Setup Wizard tương tác (chọn bàn cờ, chế độ, cấp AI 1-7, ngân sách, seed)
python scripts/play.py

# 2. Chơi Người vs Máy trực tiếp (bàn 4x4 K=4, người cầm X đấu với HybridMCTS Cấp 7, 50 ms/nước)
python scripts/play.py --board 4x4_k4 --mode human_vs_ai --human-player X --agent-o 7 --budget-ms 50

# 3. Xem Máy vs Máy tự động (Live Dashboard không chớp giật: L7 HybridMCTS vs L3 AlphaBeta)
python scripts/play.py --board 4x4_k4 --mode ai_vs_ai --agent-x 7 --agent-o 3 --budget-ms 50 --step-delay 600

# 4. Chế độ Thuyết trình Từng Bước (nhấn Enter trước mỗi nước đi của AI để giảng giải heatmap)
python scripts/play.py --board 4x4_k4 --mode ai_vs_ai --agent-x 7 --agent-o 4 --step-by-step

# 5. Bật chế độ ASCII thuần cho Windows CMD / terminal legacy không hỗ trợ UTF-8 box-drawing
python scripts/play.py --board 4x4_k4 --mode human_vs_ai --difficulty hard --ascii
```

> Lưu ý: Mức `--difficulty easy / medium / hard` hoặc số cấp `1..7` đều được hỗ trợ. Riêng Cấp 5, 6, 7 chỉ khả dụng khi checkpoint đã huấn luyện và file `*.meta.json` tương ứng tồn tại hợp lệ trong `checkpoints/` (tránh chạy mô hình chưa huấn luyện).

### Cách đọc Dashboard & Thao tác trong trận

Giao diện Live Match Dashboard chia làm **2 cột cố định (Trái 48% / Phải 52%)** theo [Planning/DESIGN.md](./Planning/DESIGN.md):
- **Cột trái — Game Board:**
  - Quân **`X`** tô màu **Bright Cyan**, quân **`O`** tô màu **Bright Red**.
  - Nước vừa đánh được đánh dấu bằng ngoặc vuông **`[ ]`** trên nền xám nổi bật; khi kết thúc ván thắng, toàn bộ **`K` ô đường thắng** được tô nền **Bright Green**.
- **Cột phải — AI Telemetry & Action Heatmap:**
  - **AI Telemetry:** Hiển thị nước chọn, độ trễ suy nghĩ (`ms`, tự đổi màu theo tỷ lệ dùng ngân sách), số `Nodes Visited` / `Simulations`, `Depth Reached`, và `Value Estimate`.
  - **Action Heatmap 5 mức màu & Bảng Top-5 Moves:** Chuyển đổi `action_scores` thành thang nhiệt trực quan (`Bright Yellow` Top-1 $\rightarrow$ `Green` $\rightarrow$ `Yellow` $\rightarrow$ `Dim White`). Riêng với **Cấp 7 (`HybridMCTS`)**, bảng Top-5 hiển thị **song song** cả xác suất tiên nghiệm **`Prior P(s,a)`** (gợi ý ban đầu của mạng Policy) và **`Visit Share N(s,a)/N(s)`** (phân phối sau khi cây PUCT rà soát), minh họa trực tiếp cách tìm kiếm cây điều chỉnh hoặc củng cố gợi ý mạng nơ-ron (**RQ1**).
- **Footer & Lệnh điều khiển (khi chơi có Người):**
  - Nhập nước đi dạng tọa độ **`r,c`** (ví dụ `1,2`), **`r c`**, hoặc **chỉ số ô `0..MN-1`** (ví dụ `6`).
  - Lệnh hỗ trợ: **`u`** / **`undo`** (hoàn tác 2 lượt để thử lại thế cờ), **`h`** / **`help`** (xem hướng dẫn), **`q`** / **`quit`** (thoát ván mượt mà). Nhập sai hoặc chọn ô đã đánh sẽ hiển thị cảnh báo vàng ngay trên Footer mà không làm gián đoạn ván đấu.

---

## 8. Minh bạch Học thuật & Kế thừa

Tuân thủ nguyên tắc minh bạch học thuật tại [Planning/RULES.md](./Planning/RULES.md), dự án phân định rõ ràng ranh giới giữa **phần tự thiết kế/cài đặt 100%** và **phần kế thừa/tham khảo khái niệm** từ thư mục `References/`:

| Nguồn trong `References/` | Phần Kế thừa / Tham khảo Khái niệm (*Adapted / Inspired*) | Phần Tự Thiết kế & Cài đặt Mới 100% (*Original Contribution*) |
|---|---|---|
| **`Tic tac toe Game.pdf`** *(Slide thiết kế)* | Kiến trúc mạng đối chứng `FastPolicyNet` (2 lớp `64`), `StrongPolicyNet` (`128` conv + 3 lớp `64`, slide 44–48), `ValueNet` (`Tanh`, slide 61–62), biểu diễn bàn cờ chuẩn hóa góc nhìn người chơi hiện tại (`canonical_board`, slide 38), bảng luật Newell & Simon và Tabular Q trên `3×3` (slide 56). | Tổng quát hóa từ `3×3` sang **`M×N, K`** với PyTorch `(B, 1, M, N)` (`padding=1`), cơ chế **Action Masking (`-1e9`)** dùng chung, bộ luật `RuleBasedAgent` 6 tầng phân giải xung đột tổng quát, **Iterative Deepening** ngắt theo thời gian thực (`time.perf_counter()`), lọc trùng & tách dữ liệu **theo `game_id`**, **Double-DQN Self-Play + Mixed Pool** cho `4×4`, và **`HybridMCTSAgent`** kèm LRU Prior Cache. |
| **`2024 - Liu - AlphaGo Simplified...pdf`** & **`AlphaGoSimplified-main/`** *(Mark Liu, 2024 — chỉ đọc đối chiếu)* | Lộ trình tiến hóa thuật toán từ Rule-Based $\rightarrow$ Minimax/Alpha-Beta $\rightarrow$ MCTS UCT $\rightarrow$ Deep Policy/Value Network $\rightarrow$ RL $\rightarrow$ mô hình lai AlphaGo-inspired. | **Viết mới 100% bằng PyTorch & Python hiện đại** (không sao chép mã TensorFlow/Keras từ notebook gốc); xây dựng giao ước `BaseAgent`/`AgentDecision`, bộ kiểm tra metadata/SHA-256 chống nạp sai hoặc chưa train (`checkpoint_io.py`), và bộ thống kê Wilson / Bootstrap CI (`metrics.py`). |
| **`Reversi-mcts-master/`** *(Kho mã tham khảo C# — chỉ đọc đối chiếu)* | Tham khảo cấu trúc 4 pha Selection – Expansion – Simulation – Backpropagation của MCTS UCT và quản lý thống kê nút cây. | Cài đặt thuần Python/NumPy tối ưu bộ nhớ (`_MCTSNode` & `_PUCTNode` dùng `__slots__`, giải phóng cây sau nước đi), cơ chế giới hạn song song theo `n_simulations` lẫn `time_budget_ms`, và mở rộng thành cây **PUCT lai Policy/Value**. |
| **`Artificial Intelligence A Modern Approach 4th.pdf`** & **`SuttonBartoIPRLBook2ndEd.pdf`** | Cơ sở lý thuyết về Minimax, Alpha-Beta Pruning, Move Ordering (*AIMA* 4th ed.) và Tabular Q-learning / Bellman optimality / Experience Replay (*Sutton & Barto* 2nd ed.). | Hiện thực hóa trực tiếp trên môi trường `ExtendedTicTacToe` tổng quát kèm kiểm chứng đẳng thức toán học (`Minimax == AlphaBeta` ở cùng độ sâu) và thực nghiệm đa seed. |

---

## 9. Bảng Tự Đánh giá Ánh xạ Rubric (100 Điểm + 10 Điểm Khuyến khích)

Đối chiếu đầy đủ 8 tiêu chí chính thức và phần Điểm Khuyến khích từ [Planning/PRD.md](./Planning/PRD.md) (Mục 6) với bằng chứng kiểm chứng cụ thể trong repository:

| STT | Tiêu chí Rubric | Điểm tối đa | Minh chứng Cụ thể trong Repository | Trạng thái |
|---|---|---:|---|---|
| **1** | **Xác định bài toán / game** | **10** | Định nghĩa MDP & luật tổng quát `M×N, K` trong [PRD.md](./Planning/PRD.md); 3 câu hỏi nghiên cứu **RQ1–RQ3**; bộ kiểm thử Engine & Luật chơi [test_engine.py](./tests/test_engine.py), [test_rules.py](./tests/test_rules.py), [test_config.py](./tests/test_config.py) pass 100%. | Đạt chuẩn |
| **2** | **Xây dựng hệ thống** | **15** | Kiến trúc phân lớp 5 phân hệ (`engine`, `agents`, `training`, `benchmark`, `ui`) trong [src/ttt_ai/](./src/ttt_ai/); 7+ Public Game APIs (`step` kiểm tra chặt `IllegalMoveError`, `clone()` độc lập, `undo()` $O(1)$); AST test phát hiện bare `print()` & truy cập private attrs; giao diện Rich CLI [cli_app.py](./src/ttt_ai/ui/cli_app.py) (`32` CLI tests pass). | Đạt chuẩn |
| **3** | **Tác tử tìm kiếm (Cấp 1–4)** | **20** | Cấp 1 [`RandomAgent`](./src/ttt_ai/agents/random_agent.py), Cấp 2 [`RuleBasedAgent`](./src/ttt_ai/agents/rule_based_agent.py) (6 tầng ưu tiên rõ xung đột), Cấp 3 [`MinimaxAgent` / `AlphaBetaAgent`](./src/ttt_ai/agents/minimax_agent.py) (so sánh bật/tắt cắt tỉa & move ordering giảm `-94.71%` nút ở `d=4`), Cấp 4 [`MCTSAgent`](./src/ttt_ai/agents/mcts_agent.py) (4 pha UCT). Kiểm chứng qua `test_level1_level2_agents.py`, `test_minimax_agent.py`, `test_mcts_agent.py`, `test_search_agents.py`, `test_agents_contract.py`. | Đạt chuẩn |
| **4** | **Tác tử học được (Cấp 5–7)** | **15** | **15 checkpoints** đã huấn luyện thực tế + `.meta.json` trong [`checkpoints/`](./checkpoints/): `Fast/StrongPolicyNet` & `ValueNet` (`3×3`, `4×4`, tách tập theo `game_id`), `TabularQAgent` (`3×3` x 3 seeds), `DQNAgent` Self-Play (`4×4 K=4`, `4×4 K=3` x 3 seeds), và `HybridMCTSAgent` (PUCT + LRU cache + `rollout`/`mixed`). Kèm raw learning curves `logs/train_*.jsonl`. | Đạt chuẩn |
| **5** | **Thiết kế đánh giá** | **15** | Giải đấu **Paired Round-Robin** ([tournament.py](./src/ttt_ai/benchmark/tournament.py)): **200 ván/cặp** (`100` đi `X` + `100` đi `O`), **4 giải chính thức = 16,800 ván** ở cả *Default* và *Fixed Budget (`10 ms`, `50 ms`)* + đối chứng `3×3 K=3`, `0` `IllegalMoveError`. Kèm **4 kịch bản Ablation (`8,756` ván)** trong [ablations.py](./src/ttt_ai/benchmark/ablations.py). | Đạt chuẩn |
| **6** | **Kết quả & thảo luận** | **10** | Tự động tính Wilson/Bootstrap 95% CI, $\Delta\text{Score}$, `mean_ms`, `p95_ms`, `avg_nodes/sims` trong [metrics.py](./src/ttt_ai/benchmark/metrics.py); tổng hợp đủ trong [summary_tables.md](./reports/summary_tables.md), [tournament_suite_summary.md](./reports/tournament_suite_summary.md), [ablation_summary.md](./reports/ablation_summary.md), và phân tích ván lỗi thực tế [sample_failure_cases.md](./reports/sample_failure_cases.md). | Đạt chuẩn |
| **7** | **Báo cáo, slide, demo** | **10** | Pipeline [visualizer.py](./src/ttt_ai/benchmark/visualizer.py) + [plot_results.py](./scripts/plot_results.py) xuất tự động **Fig 1..5** (`300 DPI` PNG, không đè nhãn) trong [`reports/figures/`](./reports/figures/); Demo Rich CLI [play.py](./scripts/play.py) chạy mượt cả `Human vs AI` và `AI vs AI`; đầy đủ bộ 6 tài liệu `Planning/`; **Báo cáo toàn văn LaTeX** [`Reports_Project/LaTeX/main.tex`](./Reports_Project/LaTeX/main.tex) (7 chương, chuẩn SGU); **Slide thuyết trình 13 trang** [`Slide_Extended_TicTacToe.pptx`](./Reports_Project/Slide_Extended_TicTacToe.pptx); **Poster khoa học A3** [`Poster_A3_Extended_TicTacToe.pdf`](./Reports_Project/Poster_A3_Extended_TicTacToe.pdf); và **Sổ tay thực nghiệm tương tác** [`Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb`](./Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb) (22 cells, 6 phần). | Đạt chuẩn |
| **8** | **Tái lập & đạo đức** | **5** | `ruff check src/ tests/ scripts/` sạch PEP 8; **347/347 `pytest`** pass; quản lý seed đồng bộ; lưu toàn bộ raw logs `.jsonl` kèm metadata phần cứng; minh bạch đối chiếu `References/` trong README & docstrings; dữ liệu báo cáo trích xuất nguyên gốc từ log thô (*Zero Fabrication*). | Đạt chuẩn |
| **+** | **Điểm khuyến khích (Bonus)** | **+10** | Thực nghiệm đối chứng thuyết phục **Cấp 7 `HybridMCTS`** (`rollout` & `mixed` vs `Pure MCTS` vs `PolicyNet`, minh họa **RQ1**); kiểm thử đa cấu hình **`3×3 K=3`**, **`4×4 K=3`**, **`4×4 K=4`**, **`5×5 K=4`** ([game_5x5_k4.yaml](./configs/game_5x5_k4.yaml)); và **Real-time Explainability Heatmap** (hiển thị đồng thời `Prior` & `Visit Share` của Cấp 7) trên Rich CLI. | Đạt chuẩn |
| | **Tổng cộng** | **100 (+10)** | Toàn bộ 5/5 Sprint (`Phase 0` + `T1.1`–`T5.3`) đã hoàn thành & kiểm chứng. | **READY** |
