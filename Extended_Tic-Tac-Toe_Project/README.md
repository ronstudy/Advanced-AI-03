# Xây dựng và Đánh giá Game AI Đa Cấp Độ — Extended Tic-Tac-Toe (`M×N, K`)

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1.0-ee4c2c.svg)](./pyproject.toml)
[![Pytest](https://img.shields.io/badge/Pytest-347%20passed-brightgreen.svg)](#5-huong-dan-tai-lap-thuc-nghiem-1-lenh)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20%7C%20PEP%208-000000.svg)](./pyproject.toml)
[![Academic Integrity](https://img.shields.io/badge/Integrity-Zero%20Fabrication-informational.svg)](#7-minh-bach-hoc-thuat--ke-thua)

> **Hệ thống mô phỏng, huấn luyện, đánh giá đối kháng công bằng và trình diễn Rich CLI cho bài toán Game AI 7 cấp độ trên trò chơi Extended Tic-Tac-Toe tổng quát `M×N, K` (chuẩn thực nghiệm chính `4×4, K=4`).**

---

## Mục lục

1. [Tổng quan Dự án & Ba Câu hỏi Nghiên cứu (`RQ1, RQ2, RQ3`)](#1-tong-quan-du-an--ba-cau-hoi-nghien-cuu-rq1-rq2-rq3)
2. [Kiến trúc Hệ thống & 7 Cấp độ Tác tử AI](#2-kien-truc-he-thong--7-cap-do-tac-tu-ai)
3. [Cấu trúc Thư mục Dự án](#3-cau-truc-thu-muc-du-an)
4. [Hướng dẫn Cài đặt & Khởi tạo Môi trường (Quick Start)](#4-huong-dan-cai-dat--khoi-tao-moi-truong-quick-start)
5. [Hướng dẫn Tái lập Thực nghiệm 1 Lệnh (One-Command Reproducibility)](#5-huong-dan-tai-lap-thuc-nghiem-1-lenh)
6. [Hướng dẫn Trải nghiệm Trực quan (`Rich CLI` Dashboard)](#6-huong-dan-trai-nghiem-truc-quan-rich-cli-dashboard)
7. [Minh bạch Học thuật & Kế thừa (`References/`)](#7-minh-bach-hoc-thuat--ke-thua)
8. [Bảng Tự Đánh giá Ánh xạ Rubric (100 Điểm + 10 Điểm Khuyến khích)](#8-bang-tu-danh-gia-anh-xa-rubric-100-diem--10-diem-khuyen-khich)

---

## 1. Tổng quan Dự án & Ba Câu hỏi Nghiên cứu (`RQ1, RQ2, RQ3`)

Trên bàn cờ `3×3` cổ điển ($3^9 = 19,683$ cấu hình), các thuật toán tìm kiếm cạn kiệt hoặc luật heuristic mạnh dễ nhanh chóng kéo ván đấu về trạng thái hòa hoàn toàn, làm mờ ranh giới khác biệt giữa tìm kiếm cây, mạng học sâu và học tăng cường. Dự án thiết kế bộ Engine **Extended Tic-Tac-Toe (`M×N, K`)** tổng quát (trò chơi đối kháng 2 người, quan sát đầy đủ, tất định, tổng bằng không `zero-sum`) và chọn **`4×4, K=4`** ($3^{16} \approx 4.3 \times 10^7$ cấu hình) làm **chuẩn thực nghiệm chính**, kèm các cấu hình đối chứng **`4×4, K=3`**, **`3×3, K=3`**, và **`5×5, K=4`**.

Thay vì chỉ xếp hạng thắng/thua đơn thuần, hệ thống thu thập **16,800 ván giải đấu chính thức** (4 giải `Paired Round-Robin`, cân bằng 50% `X` / 50% `O`) và **8,756 ván thực nghiệm Ablation** (cùng **84 ván** smoke kiểm chứng `T4.1`) để trả lời có căn cứ thống kê **3 câu hỏi nghiên cứu (RQ1–RQ3)**:

| Mã | Câu hỏi Nghiên cứu Trọng tâm | Kết luận Thực nghiệm Tóm tắt (từ raw `logs/*.jsonl` & [summary_tables.md](./reports/summary_tables.md)) |
|---|---|---|
| **RQ1** | **Hybrid MCTS (AlphaGo-inspired) so với Pure MCTS (UCT) & Policy-only** | Tại ngân sách mô phỏng thấp (`50 sims` trên `4×4, K=4`), `HybridRollout_sim50` (`0.583 [0.542, 0.625]`, chỉ `5` thua) và `HybridMixed_sim50` (`0.575 [0.533, 0.621]`, độ sâu `6.46`) vượt trội cả `PureMCTS_sim50` (`0.529 [0.471, 0.588]`, `22` thua, độ sâu `2.42`) lẫn `Policy_Strong` không tìm kiếm (`0.525`). Ở giới hạn thời gian CPU chặt (`10 ms` / `50 ms`), chi phí forward PyTorch làm giảm thông lượng mô phỏng (`145.9` vs `190.7 sims` ở `10 ms`), nhưng bộ đệm **LRU Cache theo hash bàn cờ** đạt tỷ lệ trúng **`60.7%–69.0%`**. Trên `3×3, K=3` (`default`), `L7_HybridMCTS` dẫn đầu toàn giải (`0.637 [0.624, 0.650]`, `328-872-0`, `0` thua). |
| **RQ2** | **Lợi thế người đi trước ($\Delta\text{Score} = \text{Score}_X - \text{Score}_O$) theo luật thắng `K` & cấp AI** | Khi nới bàn cờ thành `4×4` nhưng giữ `K=3`, lợi thế đi trước bùng nổ với `X Win = 76.5%`, `Draw = 0.0%`, $\Delta\text{Score} = +0.530\;[+0.445, +0.610]$ (`RuleBased`, `AlphaBeta`, `DQN` đạt `1.000` khi đi `X`). Tăng chuỗi thắng lên `4×4, K=4` mở rộng không gian phòng thủ, đẩy tỷ lệ hòa lên `74.1%` (`71.1%–73.3%` trong các giải `4,200` ván) và kéo $\Delta\text{Score}$ về gần cân bằng $+0.024\;[-0.011, +0.058]$ (so với $+0.112\;[+0.073, +0.152]$ trên `3×3, K=3`). |
| **RQ3** | **Độ ổn định đa seed & Tổng quát hóa: RL Self-Play vs Supervised Learning** | Khi gặp các đối thủ cây tìm kiếm chưa từng xuất hiện lúc huấn luyện (*Unseen Opponents*: `MCTS`, `AlphaBeta`), tác tử RL (`DQNAgent` với Mixed Pool 70% Self-Play / 30% Baseline trên `4×4, K=4`, `3 seeds {42, 123, 999}`) đạt **`0.586 ± 0.027`** tổng thể và **`0.554 ± 0.016`** trước Unseen `MCTS` (chỉ `2–3` thua / 360 ván). Ngược lại, `SL_StrongPolicy` giảm xuống **`0.497`** (`0.344` trước `MCTS`, `30` thua) và `SL_ValueGreedy` tụt xuống **`0.307`** (`0.256` trước `MCTS`). Trên `3×3, K=3`, `TabularQ` 3 seed đạt **`0.659 ± 0.005`** (`0.619 ± 0.023` trước `MCTS`) so với `0.469–0.485` của nhóm Supervised. |

---

## 2. Kiến trúc Hệ thống & 7 Cấp độ Tác tử AI

Hệ thống áp dụng kiến trúc phân tầng lỏng (*loosely coupled architecture*) chia thành 5 gói độc lập: `engine`, `agents`, `training`, `benchmark`, và `ui`.

- **Hợp đồng Game API nghiêm ngặt (`ExtendedTicTacToe`):** Mọi tác tử AI chỉ được phép tương tác thông qua 7+ phương thức công khai: `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result(player)`, `render()`, `clone()`, và `undo()`. Tuyệt đối không truy cập biến nội bộ `_board`/`_history` (được kiểm tra tự động bằng AST tests trong `tests/`).
- **Chống rò rỉ & Chống dùng trọng số chưa train:** Dữ liệu Supervised chia `train / val / test = 70% / 15% / 15%` theo **`game_id`**; mọi checkpoint `.pt`/`.pkl` đều đi kèm file metadata `*.meta.json` chứa hash SHA-256 và `board_config` để kiểm chứng hợp lệ khi nạp.

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

> **Minh bạch phạm vi Cấp 7:** `HybridMCTSAgent` được định vị là mô hình **AlphaGo-inspired** (Supervised `PolicyNet` + `ValueNet` kết hợp cây PUCT trên CPU), **không** tuyên bố tái hiện kiến trúc phân tán đa GPU/TPU hay vòng lặp tự học ResNet hoàn chỉnh của AlphaZero nguyên bản.

---

## 3. Cấu trúc Thư mục Dự án

```text
Extended_Tic-Tac-Toe_Project/
├── Planning/                   # Bộ 6 tài liệu thiết kế, quy chuẩn & nhật ký tiến độ
│   ├── PRD.md                  # Đặc tả 3 RQ, MDP, 7 cấp AI & Rubric 100+10đ
│   ├── RULES.md                # Chuẩn PEP 8, Zero Fabrication & Hợp đồng Game API
│   ├── ARCHITECTURE.md         # Kiến trúc phân lớp, schema JSONL & checkpoint metadata
│   ├── DESIGN.md               # Thiết kế Rich CLI Dashboard & Fig 1..5
│   ├── TASKS.md                # Phân rã công việc Sprint 1..5
│   └── PROGRESS.md             # Nhật ký kiểm chứng, Bug Log (BUG-01..04) & Experiment Registry
├── References/                 # Đề cương, slide, sách Mark Liu (2024) & kho mã đối chiếu (chỉ đọc)
│   ├── TTNTNC_De_tai_cuoi_ky_Game_AI.pdf
│   ├── Tic tac toe Game.pdf
│   ├── 2024 - Liu - AlphaGo Simplified Rule-Based AI and Deep Learning in Everyday Games.pdf
│   ├── Artificial Intelligence A Modern Approach 4th.pdf
│   ├── SuttonBartoIPRLBook2ndEd.pdf
│   ├── AlphaGoSimplified-main/
│   └── Reversi-mcts-master/
├── configs/                    # Cấu hình YAML tập trung (Zero Magic Numbers)
│   ├── game_{3x3_k3,4x4_k3,4x4_k4,5x5_k4}.yaml
│   ├── train_supervised.yaml
│   ├── train_rl.yaml
│   └── tournament.yaml
├── src/ttt_ai/                 # Gói thư viện lõi (editable package)
│   ├── config.py               # Typed Dataclasses, YAML loader & seed_everything()
│   ├── engine/                 # rules.py, exceptions.py, game.py (ExtendedTicTacToe)
│   ├── agents/                 # base.py + 7 cấp AI (nn_models.py, hybrid_mcts_agent.py, ...)
│   ├── training/               # dataset_gen.py, train_supervised.py, train_q_tabular.py, train_dqn_selfplay.py, checkpoint_io.py
│   ├── benchmark/              # tournament.py, metrics.py, ablations.py, visualizer.py
│   └── ui/                     # theme.py, cli_app.py (InteractiveCLIApp + Setup Wizard + Heatmap)
├── scripts/                    # 6 CLI entrypoints tái lập thực nghiệm & demo
│   ├── generate_dataset.py
│   ├── train_all.py
│   ├── run_tournament.py
│   ├── run_ablations.py
│   ├── plot_results.py
│   └── play.py
├── tests/                      # 18 module kiểm thử tự động (347 test cases pass 100%)
├── checkpoints/                # 15 trọng số đã huấn luyện (.pt/.pkl) + 15 file *.meta.json
├── logs/                       # Raw JSONL logs: giải đấu (16,884 ván), ablations (8,756 ván), train curves & datasets
├── reports/                    # Biểu đồ 300 DPI (figures/fig1..5*.png) + CSV/MD reports
├── pyproject.toml
└── README.md
```

---

## 4. Hướng dẫn Cài đặt & Khởi tạo Môi trường (Quick Start)

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

## 5. Hướng dẫn Tái lập Thực nghiệm 1 Lệnh

Mọi bước kiểm thử, sinh dữ liệu, huấn luyện, đấu giải, ablation và vẽ biểu đồ đều có thể chạy lại chỉ bằng **một lệnh duy nhất** từ thư mục `Extended_Tic-Tac-Toe_Project/`:

### 5.1. Kiểm thử Đơn vị, Tích hợp & Kiểm tra Phong cách Code
```bash
# Chạy toàn bộ 347 test cases (engine, 7 agents, AST rules, training, tournament, metrics, ablations, CLI, visualizer)
pytest -v

# Kiểm tra PEP 8, độ dài dòng <= 100 và import sorting với Ruff
ruff check src/ tests/ scripts/
```

### 5.2. Sinh dữ liệu Chuyên gia & Huấn luyện Cấp 5 (Supervised Learning)
Dữ liệu được sinh từ `AlphaBetaAgent` kết hợp `RuleBasedAgent` (có nhiễu mở đầu) và chia `70% / 15% / 15%` theo **`game_id`** để chống rò rỉ dữ liệu:
```bash
# Sinh toàn bộ dataset (4x4_k4 & 3x3_k3) VÀ huấn luyện đủ FastPolicyNet, StrongPolicyNet, ValueNet
python scripts/generate_dataset.py --board all --train

# Hoặc chạy riêng cho bàn cờ chính 4x4, K=4:
python scripts/generate_dataset.py --board 4x4_k4 --train --seed 42
```

### 5.3. Huấn luyện Cấp 6 Đa Seed (Reinforcement Learning)
Script [scripts/train_all.py](./scripts/train_all.py) điều phối trực tiếp hai mô-đun [train_q_tabular.py](./src/ttt_ai/training/train_q_tabular.py) (`TabularQAgent` trên `3×3 K=3`) và [train_dqn_selfplay.py](./src/ttt_ai/training/train_dqn_selfplay.py) (`DQNAgent` Self-Play + Mixed Opponent Pool trên `4×4 K=4` & `4×4 K=3`) qua bộ 3 seeds `[42, 123, 999]`:
```bash
# Huấn luyện đầy đủ TabularQ (3x3) và DQN Self-Play (4x4_k4, 4x4_k3) qua cả 3 seeds (42, 123, 999)
python scripts/train_all.py --mode all --board all

# Hoặc chạy tách riêng từng nhánh:
python scripts/train_all.py --mode tabular --board 3x3_k3 --seeds 42 123 999
python scripts/train_all.py --mode dqn --board 4x4_k4 --seeds 42 123 999
```

### 5.4. Chạy Giải đấu Paired Round-Robin Chính thức (16,800 ván)
```bash
# Chạy đủ bộ 4 giải đấu chính thức (4x4 default, fixed_10ms, fixed_50ms & 3x3 default; 200 ván/cặp)
python scripts/run_tournament.py --suite

# Kiểm tra tính toàn vẹn & tái tổng hợp bảng báo cáo từ logs có sẵn (không chạy lại trận):
python scripts/run_tournament.py --suite --verify-only

# Hoặc chạy 1 giải đơn lẻ (ví dụ 4x4 K=4, giới hạn 10 ms/nước):
python scripts/run_tournament.py --board 4x4_k4 --budget-mode fixed_10ms --n-pairs 100 --seed 42
```

### 5.5. Chạy 4 Thí nghiệm Ablation Khoa học (`AB Pruning`, `RQ1`, `RQ2`, `RQ3`)
```bash
# Chạy trọn bộ 4 kịch bản ablation ở quy mô đầy đủ (8,756 ván) và xuất báo cáo ablation_*.{csv,md}
python scripts/run_ablations.py --ablation all --mode full --seed 42

# Chế độ chạy thử nhanh (smoke check) cho 1 kịch bản cụ thể (ví dụ RQ1):
python scripts/run_ablations.py --ablation 2 --mode smoke --seed 42
```

### 5.6. Kết xuất Tự động Biểu đồ (`Fig 1..5`, 300 DPI) & Báo cáo Thực nghiệm
Đọc trực tiếp từ raw `logs/*.jsonl` (tuyệt đối không hardcode số liệu) để xuất toàn bộ 5 hình PNG `300 DPI` và 3 file báo cáo định lượng (`reports/summary_metrics.csv`, `reports/summary_tables.md`, `reports/sample_failure_cases.md`):
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

## 6. Hướng dẫn Trải nghiệm Trực quan (`Rich CLI` Dashboard)

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

> Lưu ý: Mức `--difficulty easy / medium / hard` hoặc số cấp `1..7` đều được hỗ trợ. Riêng Cấp 5, 6, 7 chỉ khả dụng khi checkpoint đã huấn luyện và file `*.meta.json` tương ứng tồn tại hợp lệ trong `checkpoints/` (ngăn chặn chạy mô hình chưa train).

### Cách đọc Dashboard & Thao tác trong trận

Giao diện Live Match Dashboard chia làm **2 cột cố định (Trái 48% / Phải 52%)** theo [DESIGN.md](./Planning/DESIGN.md):
- **Cột trái — Game Board:**
  - Quân **`X`** tô màu **Bright Cyan**, quân **`O`** tô màu **Bright Red**.
  - Nước vừa đánh được đánh dấu bằng ngoặc vuông **`[ ]`** trên nền xám nổi bật; khi kết thúc ván thắng, toàn bộ **`K` ô đường thắng** được tô nền **Bright Green**.
- **Cột phải — AI Telemetry & Action Heatmap:**
  - **AI Telemetry:** Hiển thị nước chọn, độ trễ suy nghĩ (`ms`, tự đổi màu theo tỷ lệ dùng ngân sách), số `Nodes Visited` / `Simulations`, `Depth Reached`, và `Value Estimate`.
  - **Action Heatmap 5 mức màu & Bảng Top-5 Moves:** Chuyển đổi `action_scores` thành thang nhiệt trực quan (`Bright Yellow` Top-1 $\rightarrow$ `Green` $\rightarrow$ `Yellow` $\rightarrow$ `Dim White`). Riêng với **Cấp 7 (`HybridMCTS`)**, bảng Top-5 hiển thị **song song** cả xác suất tiên nghiệm **`Prior P(s,a)`** (gợi ý ban đầu của mạng Policy) và **`Visit Share N(s,a)/N(s)`** (phân phối sau khi cây PUCT rà soát), minh họa trực tiếp cách tìm kiếm cây sửa sai hoặc củng cố gợi ý mạng nơ-ron (**RQ1**).
- **Footer & Lệnh điều khiển (khi chơi có Người):**
  - Nhập nước đi dạng tọa độ **`r,c`** (ví dụ `1,2`), **`r c`**, hoặc **chỉ số ô `0..MN-1`** (ví dụ `6`).
  - Lệnh hỗ trợ: **`u`** / **`undo`** (hoàn tác 2 lượt để người chơi thử lại thế cờ), **`h`** / **`help`** (xem hướng dẫn), **`q`** / **`quit`** (thoát ván mượt mà). Nhập sai hoặc chọn ô đã đánh sẽ hiển thị cảnh báo vàng, **tuyệt đối không crash chương trình**.

---

## 7. Minh bạch Học thuật & Kế thừa

Tuân thủ nguyên tắc trung thực học thuật tại [Planning/RULES.md](./Planning/RULES.md), nhóm tác giả phân định rõ ràng ranh giới giữa **phần tự thiết kế/cài đặt 100%** và **phần kế thừa/tham khảo khái niệm** từ thư mục `References/`:

| Nguồn trong `References/` | Phần Kế thừa / Tham khảo Khái niệm (*Adapted / Inspired*) | Phần Nhóm Tự Thiết kế & Cài đặt Mới 100% (*Original Contribution*) |
|---|---|---|
| **`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`** *(Đề cương môn học)* | Khung yêu cầu 7–8 cấp AI, 7 phương thức chuẩn của Game API (`reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`), thể thức Paired Round-Robin, đo `p95_ms`, Fixed Compute Budget vs Default, và các hạng mục Rubric. | Chọn bài toán `Extended Tic-Tac-Toe (M×N, K)` với chuẩn `4×4 K=4`; phát biểu và thiết kế bộ 4 thí nghiệm ablation trả lời **RQ1, RQ2, RQ3**; cài đặt trọn vẹn codebase `src/ttt_ai/`, 18 file test (`347` cases) và Rich CLI Dashboard. |
| **`Tic tac toe Game.pdf`** *(Slide thiết kế)* | Kiến trúc mạng đối chứng `FastPolicyNet` (2 lớp `64`), `StrongPolicyNet` (`128` conv + 3 lớp `64`, slide 44–48), `ValueNet` (`Tanh`, slide 61–62), biểu diễn bàn cờ chuẩn hóa góc nhìn người chơi hiện tại (`canonical_board`, slide 38), bảng luật Newell & Simon và Tabular Q trên `3×3` (slide 56). | Tổng quát hóa từ `3×3` sang **`M×N, K`** với PyTorch `(B, 1, M, N)` (`padding=1`), cơ chế **Action Masking (`-1e9`)** dùng chung, bộ luật `RuleBasedAgent` 6 tầng phân giải xung đột tổng quát, **Iterative Deepening** ngắt theo thời gian thực (`time.perf_counter()`), lọc trùng & tách dữ liệu **theo `game_id`**, **Double-DQN Self-Play + Mixed Pool** cho `4×4`, và **`HybridMCTSAgent`** kèm LRU Prior Cache. |
| **`2024 - Liu - AlphaGo Simplified...pdf`** & **`AlphaGoSimplified-main/`** *(Mark Liu, 2024 — chỉ đọc đối chiếu)* | Lộ trình tiến hóa thuật toán từ Rule-Based $\rightarrow$ Minimax/Alpha-Beta $\rightarrow$ MCTS UCT $\rightarrow$ Deep Policy/Value Network $\rightarrow$ RL $\rightarrow$ mô hình lai AlphaGo-inspired. | **Viết mới 100% bằng PyTorch & Python hiện đại** (không sao chép mã TensorFlow/Keras từ notebook gốc); xây dựng giao ước `BaseAgent`/`AgentDecision`, bộ kiểm tra metadata/SHA-256 chống nạp sai hoặc chưa train (`checkpoint_io.py`), và bộ thống kê Wilson / Bootstrap CI (`metrics.py`). |
| **`Reversi-mcts-master/`** *(Kho mã tham khảo C# — chỉ đọc đối chiếu)* | Tham khảo cấu trúc 4 pha Selection – Expansion – Simulation – Backpropagation của MCTS UCT và quản lý thống kê nút cây. | Cài đặt thuần Python/NumPy tối ưu bộ nhớ (`_MCTSNode` & `_PUCTNode` dùng `__slots__`, giải phóng cây sau nước đi), cơ chế giới hạn song song theo `n_simulations` lẫn `time_budget_ms`, và mở rộng thành cây **PUCT lai Policy/Value**. |
| **`Artificial Intelligence A Modern Approach 4th.pdf`** & **`SuttonBartoIPRLBook2ndEd.pdf`** | Cơ sở lý thuyết về Minimax, Alpha-Beta Pruning, Move Ordering (*AIMA* 4th ed.) và Tabular Q-learning / Bellman optimality / Experience Replay (*Sutton & Barto* 2nd ed.). | Hiện thực hóa trực tiếp trên môi trường `ExtendedTicTacToe` tổng quát kèm kiểm chứng đẳng thức toán học (`Minimax == AlphaBeta` ở cùng độ sâu) và thực nghiệm đa seed. |

---

## 8. Bảng Tự Đánh giá Ánh xạ Rubric (100 Điểm + 10 Điểm Khuyến khích)

Đối chiếu đầy đủ 8 tiêu chí chính thức và phần Điểm Khuyến khích từ [Planning/PRD.md](./Planning/PRD.md) (Mục 6) với bằng chứng kiểm chứng cụ thể trong repository:

| STT | Tiêu chí Rubric | Điểm tối đa | Minh chứng Cụ thể trong Repository | Trạng thái |
|---|---|---:|---|---|
| **1** | **Xác định bài toán / game** | **10** | Định nghĩa MDP & luật tổng quát `M×N, K` trong [PRD.md](./Planning/PRD.md); 3 câu hỏi nghiên cứu **RQ1–RQ3**; bộ kiểm thử Engine & Luật chơi [test_engine.py](./tests/test_engine.py), [test_rules.py](./tests/test_rules.py), [test_config.py](./tests/test_config.py) pass 100%. | Đạt chuẩn |
| **2** | **Xây dựng hệ thống** | **15** | Kiến trúc phân lớp 5 gói (`engine`, `agents`, `training`, `benchmark`, `ui`) trong [src/ttt_ai/](./src/ttt_ai/); 7+ Public Game APIs (`step` kiểm tra chặt `IllegalMoveError`, `clone()` độc lập, `undo()` $O(1)$); AST test cấm bare `print()` & truy cập private attrs; giao diện Rich CLI [cli_app.py](./src/ttt_ai/ui/cli_app.py) (`32` CLI tests pass). | Đạt chuẩn |
| **3** | **Tác tử tìm kiếm (Cấp 1–4)** | **20** | Cấp 1 [`RandomAgent`](./src/ttt_ai/agents/random_agent.py), Cấp 2 [`RuleBasedAgent`](./src/ttt_ai/agents/rule_based_agent.py) (6 tầng ưu tiên rõ xung đột), Cấp 3 [`MinimaxAgent` / `AlphaBetaAgent`](./src/ttt_ai/agents/minimax_agent.py) (so sánh bật/tắt cắt tỉa & move ordering giảm `-94.71%` nút ở `d=4`), Cấp 4 [`MCTSAgent`](./src/ttt_ai/agents/mcts_agent.py) (4 pha UCT). Kiểm chứng qua `test_level1_level2_agents.py`, `test_minimax_agent.py`, `test_mcts_agent.py`, `test_search_agents.py`, `test_agents_contract.py`. | Đạt chuẩn |
| **4** | **Tác tử học được (Cấp 5–7)** | **15** | **15 checkpoints** đã huấn luyện thực tế + `.meta.json` trong [`checkpoints/`](./checkpoints/): `Fast/StrongPolicyNet` & `ValueNet` (`3×3`, `4×4`, tách tập theo `game_id`), `TabularQAgent` (`3×3` x 3 seeds), `DQNAgent` Self-Play (`4×4 K=4`, `4×4 K=3` x 3 seeds), và `HybridMCTSAgent` (PUCT + LRU cache + `rollout`/`mixed`). Kèm raw learning curves `logs/train_*.jsonl`. | Đạt chuẩn |
| **5** | **Thiết kế đánh giá** | **15** | Giải đấu **Paired Round-Robin** ([tournament.py](./src/ttt_ai/benchmark/tournament.py)): **200 ván/cặp** (`100` đi `X` + `100` đi `O`), **4 giải chính thức = 16,800 ván** ở cả *Default* và *Fixed Budget (`10 ms`, `50 ms`)* + đối chứng `3×3 K=3`, `0` `IllegalMoveError`. Kèm **4 kịch bản Ablation (`8,756` ván)** trong [ablations.py](./src/ttt_ai/benchmark/ablations.py). | Đạt chuẩn |
| **6** | **Kết quả & thảo luận** | **10** | Tự động tính Wilson/Bootstrap 95% CI, $\Delta\text{Score}$, `mean_ms`, `p95_ms`, `avg_nodes/sims` trong [metrics.py](./src/ttt_ai/benchmark/metrics.py); tổng hợp đủ trong [summary_tables.md](./reports/summary_tables.md), [tournament_suite_summary.md](./reports/tournament_suite_summary.md), [ablation_summary.md](./reports/ablation_summary.md), và phân tích ván lỗi thực tế [sample_failure_cases.md](./reports/sample_failure_cases.md). | Đạt chuẩn |
| **7** | **Báo cáo, slide, demo** | **10** | Pipeline [visualizer.py](./src/ttt_ai/benchmark/visualizer.py) + [plot_results.py](./scripts/plot_results.py) xuất tự động **Fig 1..5** (`300 DPI` PNG, không đè nhãn) trong [`reports/figures/`](./reports/figures/); Demo Rich CLI [play.py](./scripts/play.py) chạy mượt cả `Human vs AI` và `AI vs AI`; đầy đủ tài liệu `Planning/`. | Đạt chuẩn |
| **8** | **Tái lập & đạo đức** | **5** | `ruff check src/ tests/ scripts/` sạch PEP 8; **347/347 `pytest`** pass; quản lý seed đồng bộ; lưu toàn bộ raw logs `.jsonl` kèm metadata phần cứng; minh bạch đối chiếu `References/` trong README & docstrings; tuân thủ tuyệt đối *Zero Fabrication*. | Đạt chuẩn |
| **+** | **Điểm khuyến khích (Bonus)** | **+10** | Thực nghiệm đối chứng thuyết phục **Cấp 7 `HybridMCTS`** (`rollout` & `mixed` vs `Pure MCTS` vs `PolicyNet`, minh họa **RQ1**); kiểm thử đa cấu hình **`3×3 K=3`**, **`4×4 K=3`**, **`4×4 K=4`**, **`5×5 K=4`** ([game_5x5_k4.yaml](./configs/game_5x5_k4.yaml)); và **Real-time Explainability Heatmap** (hiển thị đồng thời `Prior` & `Visit Share` của Cấp 7) trên Rich CLI. | Sẵn sàng |
| | **Tổng cộng** | **100 (+10)** | Toàn bộ 5/5 Sprint (`Phase 0` + `T1.1`–`T5.3`) đã hoàn thành & nghiệm thu. | **READY** |
