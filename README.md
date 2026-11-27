# Advanced-AI-03 — Trí Tuệ Nhân Tạo Nâng Cao (Advanced AI Lab & Projects)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Tests-347%20passed-brightgreen?logo=pytest&logoColor=white" alt="Pytest Status" />
  <img src="https://img.shields.io/badge/Code%20Style-Ruff%20%7C%20PEP%208-black?logo=ruff&logoColor=white" alt="Code Style" />
  <img src="https://img.shields.io/badge/Capstone-5%2F5%20Sprints%20Done-success" alt="Capstone Status" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</p>

> Kho lưu trữ mã nguồn, bài tập lý thuyết, ôn tập giữa kỳ, bài thực hành Lab và đồ án cuối kỳ Game AI môn **Trí Tuệ Nhân Tạo Nâng Cao**. Repository bao trùm các chủ đề nền tảng và nâng cao của AI cổ điển đến hiện đại: Tìm kiếm không gian trạng thái (Uninformed/Informed Search, A\*, Local Search), Bài toán thỏa mãn ràng buộc (CSP), Suy luận xác suất & Mạng Bayes (Bayesian Networks, Decision Theory, Naive Bayes), Lập kế hoạch (STRIPS / Goal Stack Planning), Mạng nơ-ron (MLP, CNN), Lý thuyết trò chơi đối kháng (Minimax, Alpha-Beta Pruning, MCTS UCT), Học tăng cường (Tabular Q-Learning, Double DQN Self-Play) và Mô hình lai AlphaGo-inspired (Hybrid MCTS PUCT).

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

## 📌 Mục lục (Table of Contents)

- [Advanced-AI-03 — Trí Tuệ Nhân Tạo Nâng Cao (Advanced AI Lab & Projects)](#advanced-ai-03--trí-tuệ-nhân-tạo-nâng-cao-advanced-ai-lab--projects)
  - [Thông tin chung](#thông-tin-chung)
  - [📌 Mục lục (Table of Contents)](#-mục-lục-table-of-contents)
  - [🗂️ Cấu trúc Repository (Project Structure)](#️-cấu-trúc-repository-project-structure)
  - [🔬 Nội dung & Thuật toán triển khai (Core Topics & Algorithms)](#-nội-dung--thuật-toán-triển-khai-core-topics--algorithms)
    - [1. Bài tập Lý thuyết (Theory Exercises)](#1-bài-tập-lý-thuyết-theory-exercises)
    - [2. Bài tập & Báo cáo Giữa kỳ (Midterm)](#2-bài-tập--báo-cáo-giữa-kỳ-midterm)
    - [3. Thực hành Lab (Practical Labs)](#3-thực-hành-lab-practical-labs)
    - [4. Đồ án Cuối kỳ: Game AI Đa Cấp Độ (Capstone Project)](#4-đồ-án-cuối-kỳ-game-ai-đa-cấp-độ-capstone-project)
  - [⚙️ Cài đặt & Môi trường (Prerequisites & Installation)](#️-cài-đặt--môi-trường-prerequisites--installation)
    - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
    - [Cài đặt môi trường ảo & Thư viện](#cài-đặt-môi-trường-ảo--thư-viện)
  - [🚀 Hướng dẫn chạy (Usage & Execution)](#-hướng-dẫn-chạy-usage--execution)
    - [1. Chạy các bài thực hành Lab & Sổ tay Đồ án (Jupyter Notebooks)](#1-chạy-các-bài-thực-hành-lab--sổ-tay-đồ-án-jupyter-notebooks)
    - [2. Chạy kiểm thử tự động (Unit & Integration Tests)](#2-chạy-kiểm-thử-tự-động-unit--integration-tests)
    - [3. Chạy linter kiểm tra chuẩn mã nguồn](#3-chạy-linter-kiểm-tra-chuẩn-mã-nguồn)
    - [4. Tái lập Thực nghiệm & Khởi chạy Rich CLI Đồ án](#4-tái-lập-thực-nghiệm--khởi-chạy-rich-cli-đồ-án)
  - [📊 Kết quả & Đánh giá (Experiments & Evaluation)](#-kết-quả--đánh-giá-experiments--evaluation)
    - [1. Thực nghiệm Tìm kiếm cục bộ (Lab 03)](#1-thực-nghiệm-tìm-kiếm-cục-bộ-lab-03)
    - [2. Thực nghiệm Bài toán thỏa mãn ràng buộc (Lab 04)](#2-thực-nghiệm-bài-toán-thỏa-mãn-ràng-buộc-lab-04)
    - [3. Thực hành Tìm kiếm đối kháng & Trò chơi (Lab 05)](#3-thực-hành-tìm-kiếm-đối-kháng--trò-chơi-lab-05)
    - [4. Kết quả Chính thức Đồ án Cuối kỳ Extended Tic-Tac-Toe (5/5 Sprint)](#4-kết-quả-chính-thức-đồ-án-cuối-kỳ-extended-tic-tac-toe-55-sprint)
  - [📚 Nguồn Tham khảo & Kế thừa (References & Inheritance)](#-nguồn-tham-khảo--kế-thừa-references--inheritance)

---

## 🗂️ Cấu trúc Repository (Project Structure)

```text
Advanced-AI-03/
├── README.md                                              # Tài liệu tổng quan toàn bộ repository
├── .agents/                                               # Cấu hình môi trường tự động hóa cloud
│   └── environment.json                                   # Script cài đặt venv, PyTorch CPU, ttt-ai[dev], Jupyter, Numba
├── Ex/                                                    # Bài tập lý thuyết (Bản nộp PDF)
│   ├── ex01_AISearch_TruongChiLong_3123410198.pdf         # BT01: Tìm kiếm không gian trạng thái & Heuristic
│   ├── ex02_CSPGame_TruongChiLong_3123410198.pdf          # BT02: CSP, Minimax, Alpha-Beta & Expectiminimax
│   ├── ex03_AIGame_TruongChiLong_3123410198.pdf           # BT03: Game đối kháng nâng cao, Expectimax & MDP
│   └── BT06_Bayes_TruongChiLong_3123410198.pdf            # BT06: Suy luận xác suất, Mạng Bayes, MEU & Naive Bayes
├── Midterm/                                               # Bài tập & đề ôn tập giữa kỳ (Bản nộp PDF)
│   ├── M01 - Bai tap_TruongChiLong_3123410198.pdf         # M01: Thiết kế MLP, Backpropagation, SGD, Softmax & Cross-Entropy
│   └── M02 - Bai tap_TruongChiLong_3123410198.pdf         # M02: Ôn tập tổng hợp (A*, Alpha-Beta, Bayes Net, Blocks World, Naive Bayes)
├── LaTeX_Midterm/                                         # Mã nguồn LaTeX báo cáo tiến độ đồ án giữa kỳ
│   ├── main.tex                                           # Báo cáo tiến độ đồ án chuẩn SGU
│   └── SGU-LOGO.png                                       # Logo Trường Đại học Sài Gòn
├── Lab/                                                   # Bài thực hành lập trình, notebooks & báo cáo
│   ├── lab03_localsearch_search_TruongChiLong_3123410198/ # Lab 03: Thuật toán tìm kiếm cục bộ (Local Search)
│   │   ├── README.md                                      # Mục lục & hướng dẫn bài Lab 03
│   │   ├── BAOCAO_lab03_localsearch_search.pdf            # Báo cáo chi tiết thực nghiệm Lab 03
│   │   ├── L_maze.txt                                     # Dữ liệu ma trận mê cung chữ L
│   │   ├── maze_helper.py                                 # Tiện ích nạp bản đồ, vẽ mê cung & heuristic
│   │   ├── Maze_Example_LocalSearch.ipynb                 # Simple Hill Climbing trên mê cung chữ L
│   │   ├── n_queens.ipynb                                 # n-Queens: HC variants, SA, GA, 4 operators
│   │   ├── n_queens_fast_conflict_calculation_with_numba.ipynb # Tối ưu tính xung đột n-Queens bằng Numba JIT
│   │   ├── Simulated_Annealing_Schedules.ipynb            # Khảo sát 4 lịch giảm nhiệt độ Simulated Annealing
│   │   └── traveling_salesman_problem.ipynb               # TSP: Local Search (2-swap vs 2-opt), GA
│   ├── lab04_csp_TruongChiLong_3123410198/                # Lab 04: Bài toán thỏa mãn ràng buộc (CSP)
│   │   ├── README.md                                      # Mục lục & hướng dẫn bài Lab 04
│   │   ├── lab04_csp_TruongChiLong_3123410198.pdf         # Báo cáo thực nghiệm chi tiết Lab 04
│   │   ├── CSP_graph_coloring_example.ipynb               # CSP: Tô màu đồ thị (Backtracking, MRV, LCV, FC, Min-Conflicts)
│   │   ├── n_queens.ipynb                                 # CSP / Local Search n-Queens & hội tụ Log-Log
│   │   ├── Graph Coloring.png                             # Biểu đồ trực quan hóa đồ thị đã tô màu hợp lệ
│   │   ├── Queen.png                                      # Trực quan hóa nghiệm n-Queens
│   │   ├── Algorithm Convergence.png                      # Biểu đồ tốc độ hội tụ qua các bước lặp
│   │   └── Runtime Scalability.png                        # Đồ thị độ phức tạp thời gian theo kích thước bài toán
│   └── lab05_game_TruongChiLong_3123410198/               # Lab 05: Tìm kiếm đối kháng và Trò chơi (Adversarial Search)
│       ├── README.md                                      # Hướng dẫn cài đặt & chạy bài Lab 05
│       ├── tictactoe.py                                   # Module logic & tiện ích trò chơi Tic-Tac-Toe
│       ├── tictactoe_definitions.ipynb                    # Định nghĩa bài toán & không gian trạng thái Tic-Tac-Toe
│       ├── tictactoe_and_or_tree_search.ipynb             # Giải Tic-Tac-Toe bằng AND-OR Tree Search
│       ├── tictactoe_alpha_beta_tree_search.ipynb         # Minimax kết hợp cắt tỉa Alpha-Beta
│       ├── tictactoe_heuristic_alpha_beta_tree_search.ipynb # Heuristic Alpha-Beta Search giới hạn độ sâu
│       ├── tictactoe_pure_monte_carlo_search.ipynb        # Pure Monte Carlo Search (mô phỏng ngẫu nhiên)
│       ├── tictactoe_monte_carlo_tree_search_restricted.ipynb # Monte Carlo Tree Search (MCTS) với UCB1
│       ├── tictactoe_interactive.ipynb                    # Giao diện chơi tương tác trực tiếp với Agent
│       ├── assignment_connect4.ipynb                      # Bài tập 1: Tìm kiếm đối kháng chơi Connect 4
│       ├── assignment_mean_connect4.ipynb                 # Bài tập 2: Biến thể "Mean" Connect 4
│       ├── assignment_dots_and_boxes.ipynb                # Bài tập 3: Mô hình hóa & giải thuật Dots and Boxes
│       ├── profiling_code.ipynb                           # Đo lường thời gian & profiling hiệu năng Python
│       └── store_agent_state_information.ipynb            # Cơ chế lưu trữ trạng thái agent qua các lượt
└── Extended_Tic-Tac-Toe_Project/                          # Đồ án Cuối kỳ: Game AI Đa Cấp Độ (M×N, K) — Hoàn thành 5/5 Sprint
    ├── README.md                                          # Tài liệu thuyết minh chi tiết riêng của Đồ án cuối kỳ
    ├── pyproject.toml                                     # Cấu hình cài đặt package ttt-ai, pytest & ruff
    ├── Planning/                                          # Bộ 6 sổ tay thiết kế, quy chuẩn & nhật ký (01 -> 06)
    │   ├── PRD.md                                         # 01. Yêu cầu sản phẩm, mô hình MDP, RQ1–RQ3 & Rubric
    │   ├── ARCHITECTURE.md                                # 02. Kiến trúc phân tầng 5 phân hệ, Game API & 7 cấp AI
    │   ├── DESIGN.md                                      # 03. Thiết kế Rich CLI Dashboard & 5 bộ biểu đồ Fig 1..5
    │   ├── RULES.md                                       # 04. Quy định trung thực học thuật (Zero Fabrication) & PEP 8
    │   ├── TASKS.md                                       # 05. Phân rã 5 Sprint (T1.1–T5.3) & tiêu chí nghiệm thu DoD
    │   └── PROGRESS.md                                    # 06. Nhật ký thực nghiệm (25,640 ván), Bug Log (BUG-01..04)
    ├── Reports_Project/                                   # Bộ ấn phẩm báo cáo đồ án cuối kỳ chính thức
    │   ├── Report_Extended_TicTacToe.pdf                  # Báo cáo đồ án toàn văn PDF (7 chương chuẩn SGU)
    │   ├── Slide_Extended_TicTacToe.pptx                  # Slide thuyết trình bảo vệ đồ án (13 slide, 16:9)
    │   ├── Slide_Extended_TicTacToe.pdf                   # Bản xuất PDF của slide thuyết trình
    │   ├── Poster_A3_Extended_TicTacToe.pdf               # Poster báo cáo khoa học khổ A3 ngang (PDF vector)
    │   └── LaTeX/                                         # Mã nguồn báo cáo LaTeX hoàn chỉnh (main.tex, SGU-LOGO.png, figures)
    ├── Jupiter_Notebook/                                  # Sổ tay thực nghiệm tương tác
    │   └── Extended_Tic_Tac_Toe.ipynb                     # Notebook 22 cells, 6 phần: Engine, 7 AI, Heatmap, Fig 1..5 & RQ1–3
    ├── References/                                        # Giáo trình, slide & mã nguồn tham khảo (chỉ đọc)
    │   ├── Tic tac toe Game.pdf                           # Tài liệu định hướng trò chơi Tic-Tac-Toe mở rộng
    │   ├── 2024 - Liu - AlphaGo Simplified Rule-Based AI and Deep Learning in Everyday Games.pdf
    │   ├── Artificial Intelligence A Modern Approach 4th.pdf
    │   ├── SuttonBartoIPRLBook2ndEd.pdf
    │   ├── AlphaGoSimplified-main/                        # Mã nguồn đối chiếu mô hình AlphaGo đơn giản hóa
    │   └── Reversi-mcts-master/                           # Mã nguồn tham khảo cấu trúc MCTS UCT (C#)
    ├── configs/                                           # 7 tệp cấu hình YAML cho trò chơi, huấn luyện & giải đấu
    │   ├── game_3x3_k3.yaml                               # Bàn cờ đối chứng 3×3, K=3
    │   ├── game_4x4_k3.yaml                               # Biến thể bất đối xứng 4×4, K=3 (RQ2)
    │   ├── game_4x4_k4.yaml                               # Chuẩn thực nghiệm chính 4×4, K=4
    │   ├── game_5x5_k4.yaml                               # Biến thể mở rộng 5×5, K=4 (Bonus)
    │   ├── train_supervised.yaml                          # Siêu tham số huấn luyện mạng Policy/Value (Cấp 5)
    │   ├── train_rl.yaml                                  # Siêu tham số huấn luyện Tabular Q & DQN Self-Play (Cấp 6)
    │   └── tournament.yaml                                # Thiết lập giải đấu Paired Round-Robin & ngân sách
    ├── src/ttt_ai/                                        # Mã nguồn lõi package ttt_ai (5 phân hệ)
    │   ├── __init__.py & config.py                        # Khởi tạo package, typed configs & seed_everything()
    │   ├── engine/                                        # exceptions.py, rules.py (O(K) win check), game.py (7+1 Game API)
    │   ├── agents/                                        # base.py + 7 cấp AI (Random -> HybridMCTS PUCT) & nn_models.py
    │   ├── training/                                      # dataset_gen.py, train_supervised.py, train_q_tabular.py, train_dqn_selfplay.py, checkpoint_io.py
    │   ├── benchmark/                                     # tournament.py, metrics.py (Wilson/Bootstrap CI), ablations.py, visualizer.py
    │   └── ui/                                            # theme.py, cli_app.py (Rich Live Dashboard, Setup Wizard & Heatmap)
    ├── scripts/                                           # 6 script CLI tái lập thực nghiệm 1 lệnh & demo
    │   ├── generate_dataset.py                            # Sinh dữ liệu chuyên gia (game_id split) & train Cấp 5
    │   ├── train_all.py                                   # Huấn luyện đa seed Cấp 6 (Tabular Q & Double DQN Self-Play)
    │   ├── run_tournament.py                              # Điều phối 4 giải đấu Paired Round-Robin (16,800 ván)
    │   ├── run_ablations.py                               # Chạy 4 kịch bản Ablation khoa học (8,756 ván)
    │   ├── plot_results.py                                # Kết xuất tự động Fig 1..5 (300 DPI) & bảng tổng hợp
    │   └── play.py                                        # Khởi chạy Rich CLI Dashboard (Human vs AI / AI vs AI)
    ├── tests/                                             # conftest.py + 18 module kiểm thử tự động (347 passed tests)
    ├── checkpoints/                                       # 15 cặp trọng số đã huấn luyện (.pt/.pkl) + *.meta.json (SHA-256)
    ├── logs/                                              # Datasets (.npz), 15 train logs, 4 giải đấu & 4 ablations (25,640 ván)
    └── reports/                                           # Kết quả tổng hợp CSV/MD, sample_failure_cases.md
        ├── figures/                                       # 5 biểu đồ khoa học 300 DPI (fig1..fig5)
        └── terminal_test/                                 # Ảnh chụp minh chứng kiểm thử Sprint 1–5 & demo_play
```

---

## 🔬 Nội dung & Thuật toán triển khai (Core Topics & Algorithms)

### 1. Bài tập Lý thuyết (Theory Exercises)

| STT | Bài toán / Chủ đề | Thuật toán / Kỹ thuật chính | File / Tài liệu chính | Trạng thái / Kết quả |
|:---:|---|---|---|:---:|
| **01** | Tìm kiếm & Heuristic trong Không gian trạng thái | BFS, DFS, IDS, UCS, Greedy Best-First Search, A\*; Phân tích tính *admissible* và *consistent* của heuristic (khoảng cách Manhattan, vector vận tốc); Mô hình hóa bài toán Tháp Hà Nội ($3^n$ trạng thái). | [`Ex/ex01_AISearch_TruongChiLong_3123410198.pdf`](Ex/ex01_AISearch_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** |
| **02** | Bài toán Thỏa mãn Ràng buộc & Game AI cơ bản | Nhận diện Local Optima và Shoulders trong Local Search; Cây Minimax và tỉa nhánh Alpha-Beta; Expectiminimax và ngưỡng cắt tỉa xác suất; Forward Checking, Arc Consistency (AC-3), Heuristic MRV và LCV (Ma phương 3×3, Lập lịch đường băng, Tô màu đồ thị $K_3$). | [`Ex/ex02_CSPGame_TruongChiLong_3123410198.pdf`](Ex/ex02_CSPGame_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** |
| **03** | Tìm kiếm nâng cao, Game đối kháng & Quá trình Quyết định Markov | Đánh giá cây Minimax/Alpha-Beta nhiều tầng; Cắt tỉa nhánh ngẫu nhiên trong Expectimax; So sánh chính sách Minimax vs Expectimax vs Value Iteration (MDP); Lan truyền ràng buộc AC-3 theo thứ tự topo; Trò chơi nhiều mục tiêu và hợp tác Pacman–Casper; Cập nhật động đường đi A\* khi thay đổi trọng số cạnh. | [`Ex/ex03_AIGame_TruongChiLong_3123410198.pdf`](Ex/ex03_AIGame_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** |
| **04** | Suy luận Xác suất, Mạng Bayes & Lý thuyết Quyết định (BT06) | Định lý Bayes & Định luật xác suất toàn phần; Độc lập vô điều kiện vs độc lập có điều kiện & Collider (*explaining away*); Phân rã phân phối chung trên Mạng Bayes (Alarm, WetGrass); Suy luận chính xác bằng Enumeration & độ phức tạp $O(n \cdot d^m)$; Prior Sampling vs MCMC / Gibbs Sampling; Lý thuyết quyết định & nguyên lý **MEU**, Mạng Quyết định (Decision Network); Phân loại **MAP / Naive Bayes** (rời rạc, Gaussian PDF & Laplace Smoothing). | [`Ex/BT06_Bayes_TruongChiLong_3123410198.pdf`](Ex/BT06_Bayes_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** (13/13 bài tập chi tiết) |

---

### 2. Bài tập & Báo cáo Giữa kỳ (Midterm)

| STT | Hạng mục | Nội dung & Kỹ thuật chính | File / Tài liệu chính | Trạng thái |
|:---:|---|---|---|:---:|
| **M01** | **Bài tập Giữa kỳ 1: Mạng Nơ-ron Đa tầng (MLP)** | Thiết kế kiến trúc MLP cho 3 bài toán (Hồi quy tuyến tính, Phân loại nhị phân Sigmoid/BCE, Phân loại đa lớp Softmax/Cross-Entropy) & đếm tham số; Tính tay đầy đủ 1 bước huấn luyện MLP hồi quy (Forward pass, Backpropagation tầng ra/ẩn, cập nhật trọng số SGD $\eta=0.1$, đánh giá giảm loss); Tính Softmax, Cross-Entropy Loss và gradient tầng ra $\delta^{(2)}$ cho bài toán phân loại. | [`Midterm/M01 - Bai tap_TruongChiLong_3123410198.pdf`](Midterm/M01%20-%20Bai%20tap_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** |
| **M02** | **Bài tập Giữa kỳ 2: Ôn tập Tổng hợp AI Nâng cao** | Giải chi tiết 5 bài toán tổng hợp: (1) Mô hình không gian trạng thái & tìm kiếm **A\*** bài toán 3 bình nước (8L–5L–3L); (2) Giải thuật **Minimax & Cắt tỉa Alpha-Beta** trên cây trò chơi 4 tầng; (3) Suy luận xác suất trên **Mạng Bayes** (*Cloudy–Sprinkler–Rain–WetGrass*); (4) Biểu diễn trạng thái & lập kế hoạch **Blocks World** bằng **Goal Stack Planning**; (5) Phân loại **Naive Bayes** cho bài toán *Transportation Means*. | [`Midterm/M02 - Bai tap_TruongChiLong_3123410198.pdf`](Midterm/M02%20-%20Bai%20tap_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** |
| **RP** | **Báo cáo Tiến độ Đồ án Giữa kỳ (LaTeX)** | Mã nguồn LaTeX chuẩn SGU trình bày bối cảnh mở rộng bài toán Extended Tic-Tac-Toe (`M×N, K`), 3 câu hỏi nghiên cứu `RQ1–RQ3`, kiến trúc hệ thống và kết quả các giai đoạn đầu. | [`LaTeX_Midterm/main.tex`](LaTeX_Midterm/main.tex) | **Đã hoàn thành** |

---

### 3. Thực hành Lab (Practical Labs)

| STT | Bài toán / Chủ đề | Thuật toán / Kỹ thuật chính | File / Notebook chính | Trạng thái / Kết quả |
|:---:|---|---|---|:---:|
| **01** | Tìm kiếm đường đi trong Mê cung chữ L (Lab 03) | Simple Hill Climbing với hàm Heuristic Manhattan $h(n) = \|r - r_G\| + \|c - c_G\|$; Khảo sát các thứ tự ưu tiên hướng di chuyển (`NESW`, `ESWN`, `SWNE`, `WNES`) và ngẫu nhiên hóa. | [`Lab/.../Maze_Example_LocalSearch.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Maze_Example_LocalSearch.ipynb)<br>[`Lab/.../maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) | **Đã hoàn thành**<br>Chứng minh thuật toán kẹt tại cực trị cục bộ $(5, 9)$ với $h=6$ do tường chắn chữ L. |
| **02** | Bài toán $n$-Queens bằng Tìm kiếm cục bộ (Lab 03) | Cài đặt & so sánh 5 thuật toán: Steepest-Ascent HC, Inefficient Stochastic HC, Efficient (First-choice) Stochastic HC, Random Restart HC, Simulated Annealing. Cài đặt 4 toán tử di chuyển nâng cao (`single_step`, `col_swap`, `dual_queen`, `adaptive`) và Genetic Algorithm. | [`Lab/.../n_queens.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens.ipynb)<br>[`Lab/.../BAOCAO_lab03_localsearch_search.pdf`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/BAOCAO_lab03_localsearch_search.pdf) | **Đã hoàn thành**<br>Random Restart HC đạt 100% thành công (TB ~6.8 lần restart); SA đạt 98% ($n=8$). Toán tử `move_adaptive` hội tụ nhanh nhất. |
| **03** | Khảo sát Lịch giảm nhiệt độ trong Simulated Annealing (Lab 03) | Khảo sát 4 quy luật làm lạnh $T(t)$: *Exponential* ($T = T_0 \alpha^t$), *Linear* ($T = T_0 - \beta t$), *Fast* ($T = \frac{T_0}{1 + t}$), và *Logarithmic* ($T = \frac{T_0}{1 + \ln(1+t)}$). Phân tích cán cân Exploration vs Exploitation. | [`Lab/.../Simulated_Annealing_Schedules.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Simulated_Annealing_Schedules.ipynb) | **Đã hoàn thành**<br>Quy luật Exponential với $\alpha \in [0.99, 0.995]$ đạt hiệu quả cân bằng tối ưu nhất. |
| **04** | Tối ưu hóa tính toán xung đột $n$-Queens bằng Numba JIT (Lab 03) | So sánh độ phức tạp và hiệu năng: Hàm Naive $O(n^2)$, Hàm Linear $O(n)$ dùng mảng tần suất và tổ hợp $\binom{k}{2}$, và Hàm JIT Compiled Numba (`@njit`). | [`Lab/.../n_queens_fast_conflict_calculation_with_numba.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens_fast_conflict_calculation_with_numba.ipynb) | **Đã hoàn thành**<br>Trên $n=450$, bản Numba JIT nhanh hơn bản Python thuần $O(n)$ khoảng **90.3×** và nhanh hơn Naive $O(n^2)$ hàng nghìn lần. |
| **05** | Bài toán Người du lịch TSP (Lab 03) | Áp dụng Local Search giải TSP Euclid ($n \in \{10, 20, 30, 50\}$). So sánh toán tử hoán vị `2-swap` với toán tử đảo đoạn `2-opt`; Phần mở rộng Genetic Algorithm với Order Crossover (OX). | [`Lab/.../traveling_salesman_problem.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/traveling_salesman_problem.ipynb) | **Đã hoàn thành**<br>Toán tử `2-opt` tháo gỡ triệt để các giao cắt cạnh trên mặt phẳng, cho chiều dài lộ trình ngắn hơn rõ rệt so với `2-swap`. |
| **06** | Bài toán Tô màu đồ thị CSP (Lab 04) | Biểu diễn CSP trên đồ thị Delaunay tam giác hóa phẳng (`scipy.spatial.Delaunay`). Cài đặt Backtracking Search cơ bản kết hợp kỹ thuật heuristic: **MRV** (Minimum Remaining Values), **LCV** (Least Constraining Value), **Forward Checking (FC)** và **Min-Conflicts Local Search**. | [`Lab/.../CSP_graph_coloring_example.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/CSP_graph_coloring_example.ipynb)<br>[`Lab/.../Graph Coloring.png`](Lab/lab04_csp_TruongChiLong_3123410198/Graph%20Coloring.png) | **Đã hoàn thành**<br>Với 4 màu: 100% thuật toán giải quyết trọn vẹn (TB 8–11 nút mở rộng). Với 3 màu: Min-Conflicts kẹt 0% do sắc số $\chi(G) > 3$. |
| **07** | Mô hình hóa $n$-Queens dưới góc nhìn CSP (Lab 04) | Phân tích $n$-Queens như bài toán ràng buộc với biến $X_i \in \{1..n\}$. So sánh Backtracking CSP với Local Search; Khảo sát độ phức tạp thời gian theo thang đo Log-Log (độ dốc runtime slope). | [`Lab/.../n_queens.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/n_queens.ipynb)<br>[`Lab/.../lab04_csp_TruongChiLong_3123410198.pdf`](Lab/lab04_csp_TruongChiLong_3123410198/lab04_csp_TruongChiLong_3123410198.pdf) | **Đã hoàn thành**<br>Xác định hệ số góc Log-Log: Efficient Stochastic HC ($1.92$) vs Simulated Annealing ($1.06$), chứng minh SA mở rộng tốt hơn khi $n$ lớn. |
| **08** | Tìm kiếm Đối kháng trên Tic-Tac-Toe (Lab 05) | Xây dựng môi trường `tictactoe.py` và cài đặt trọn bộ các giải thuật tìm kiếm đối kháng kinh điển: **AND-OR Tree Search** (hành động bất định), **Minimax + Alpha-Beta Pruning**, **Heuristic Alpha-Beta** giới hạn độ sâu, **Pure Monte Carlo Search**, **MCTS UCB1** và giao diện chơi tương tác. | [`Lab/lab05_game_.../README.md`](Lab/lab05_game_TruongChiLong_3123410198/README.md)<br>[`tictactoe_alpha_beta_tree_search.ipynb`](Lab/lab05_game_TruongChiLong_3123410198/tictactoe_alpha_beta_tree_search.ipynb)<br>[`tictactoe_monte_carlo_tree_search_restricted.ipynb`](Lab/lab05_game_TruongChiLong_3123410198/tictactoe_monte_carlo_tree_search_restricted.ipynb) | **Đã hoàn thành**<br>8 notebooks + module `tictactoe.py`, bao gồm đối chứng hiệu năng cắt tỉa & MCTS. |
| **09** | Bài tập Mở rộng Game AI: Connect 4 & Dots and Boxes (Lab 05) | Áp dụng tìm kiếm đối kháng, thiết kế hàm lượng giá heuristic và lưu trữ trạng thái agent (`store_agent_state_information`, `profiling_code`) cho 3 trò chơi: **Connect 4**, biến thể **"Mean" Connect 4**, và **Dots and Boxes**. | [`assignment_connect4.ipynb`](Lab/lab05_game_TruongChiLong_3123410198/assignment_connect4.ipynb)<br>[`assignment_mean_connect4.ipynb`](Lab/lab05_game_TruongChiLong_3123410198/assignment_mean_connect4.ipynb)<br>[`assignment_dots_and_boxes.ipynb`](Lab/lab05_game_TruongChiLong_3123410198/assignment_dots_and_boxes.ipynb) | **Đã hoàn thành**<br>Hoàn thiện cài đặt, đánh giá agent và profiling tốc độ thực thi. |

---

### 4. Đồ án Cuối kỳ: Game AI Đa Cấp Độ (Capstone Project)

> 📖 Xem tài liệu thuyết minh chi tiết, phân tích 3 câu hỏi nghiên cứu (`RQ1–RQ3`) và bảng ánh xạ Rubric 100 (+10) điểm tại [`Extended_Tic-Tac-Toe_Project/README.md`](Extended_Tic-Tac-Toe_Project/README.md).

Dự án xây dựng hoàn chỉnh hệ thống mô phỏng, huấn luyện, giải đấu công bằng (**25,640 ván logged**) và giao diện trình diễn **Rich CLI Dashboard** cho bài toán **Extended Tic-Tac-Toe ($M \times N, K$)** trên 4 cấu hình bàn cờ: Chuẩn chính (`4×4, K=4` — $3^{16} \approx 4.3 \times 10^7$ trạng thái), Biến thể bất đối xứng (`4×4, K=3`), Đối chứng (`3×3, K=3`), và Mở rộng (`5×5, K=4`).

#### Hệ sinh thái 7 Cấp độ Tác tử AI (Hoàn thành 5/5 Sprint)

| Cấp độ | Tác tử AI | Thuật toán / Nguyên lý hoạt động | Module triển khai chính | Trạng thái kiểm thử / Thực nghiệm |
|:---:|---|---|---|:---:|
| **L0** | **Core Engine & Rules** | Động cơ trò chơi chuẩn PEP 8 (`7+1` Public Game API cô lập trạng thái nội bộ); Kiểm tra thắng gia tăng $O(K)$; `canonical_board()`; `clone()` & `undo()` $O(1)$; Kiểm tra tuân thủ bằng AST tests. | [`src/ttt_ai/engine/`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/engine/) | **Hoàn thành (Sprint 1)**<br>100% pass unit & contract tests. |
| **L1** | **RandomAgent** | Lựa chọn hành động ngẫu nhiên đồng đều trong không gian `legal_actions()` qua `np.random.Generator` gắn `seed`. Tác tử đối chứng baseline. | [`random_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/random_agent.py) | **Hoàn thành (Sprint 2)**<br>Độ trễ trung bình `0.03–0.06 ms/nước`. |
| **L2** | **RuleBasedAgent** | Tổng quát hóa bộ luật **Newell & Simon** 6 tầng ưu tiên cho mọi `M×N, K`: (1) Thắng ngay, (2) Chặn thua ngay, (3) Tạo fork ($\ge 2$ đường mở), (4) Chặn fork, (5) Trọng số vị trí `positional_weights()`, (6) Ngẫu nhiên có kiểm soát. | [`rule_based_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/rule_based_agent.py) | **Hoàn thành (Sprint 2)**<br>Bất bại (`0` thua) trong cả 4 giải đấu chính thức (`4,800` ván đấu); `0.07–0.13 ms/nước`. |
| **L3** | **MinimaxAgent / AlphaBetaAgent** | Tìm kiếm đối kháng Negamax (`step()/undo()` trên 1 `clone()`), hàm lượng giá cửa sổ trượt $K$ ô (`tanh`), **Alpha-Beta Pruning**, **Move Ordering**, và **Iterative Deepening** theo `time_budget_ms` (`perf_counter()`). | [`minimax_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/minimax_agent.py) | **Hoàn thành (Sprint 2)**<br>Alpha-Beta + Move Ordering **giảm 94.71% số nút duyệt** ở `d=4` trên `4×4, K=4` (từ `23,171` xuống `1,225` nút); bất bại (`0` thua) ở cả 4 giải đấu. |
| **L4** | **MCTSAgent (UCT)** | Monte Carlo Tree Search 4 pha chuẩn mực (**Selection** UCB1, **Expansion**, **Uniform Random Rollout**, **Backpropagation**); dừng theo `n_simulations` hoặc `time_budget_ms`; giải phóng bộ nhớ cây sau mỗi nước đi. | [`mcts_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/mcts_agent.py) | **Hoàn thành (Sprint 2)**<br>**Hạng 1** trên cả 3 giải `4×4, K=4`: `0.615` (`default`), `0.595` (`10ms`), `0.612` (`50ms`). |
| **L5** | **PolicyNetworkAgent / ValueGreedyAgent** | Mạng CNN+MLP PyTorch `(B, 1, M, N)` gồm `FastPolicyNet`, `StrongPolicyNet` và `ValueNet` có **Action Masking (`-1e9`)**; huấn luyện trên tập dữ liệu chuyên gia tách theo `game_id` (`70/15/15`) chống rò rỉ; xác minh checkpoint qua `*.meta.json` + SHA-256. | [`nn_models.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/nn_models.py)<br>[`policy_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/policy_agent.py)<br>[`train_supervised.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/training/train_supervised.py) | **Hoàn thành (Sprint 3)**<br>Đã huấn luyện đủ 6 checkpoints (`3×3` & `4×4`); độ trễ suy luận `0.51–1.27 ms/nước`. |
| **L6** | **TabularQAgent / DQNAgent** | Học tăng cường tự chơi (Self-Play): **Tabular Q-Learning** (canonical byte keys cho `3×3, K=3`) và **Double DQN** (Experience Replay `50,000`, Target Network, Mixed Pool 70% Self-Play / 30% Baseline cho `4×4, K=4` & `4×4, K=3`), huấn luyện đa seed `{42, 123, 999}`. | [`rl_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/rl_agent.py)<br>[`train_q_tabular.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/training/train_q_tabular.py)<br>[`train_dqn_selfplay.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/training/train_dqn_selfplay.py) | **Hoàn thành (Sprint 3)**<br>Đã huấn luyện đủ 9 checkpoints đa seed; tổng quát hóa vượt trội Supervised trước Unseen Opponents (**RQ3**). |
| **L7** | **HybridMCTSAgent** *(AlphaGo-inspired)* | MCTS biến thể **PUCT** dẫn hướng bởi prior $P(s, a)$ từ `StrongPolicyNet`, **LRU Cache theo canonical hash** (cache hit `60.7%–69.0%`), 2 chế độ đánh giá lá (`rollout` và `mixed` $\lambda V_\theta + (1-\lambda)z$); xuất đồng thời `Prior P(s,a)` & `Visit Share N(s,a)/N(s)` phục vụ Explainability. | [`hybrid_mcts_agent.py`](Extended_Tic-Tac-Toe_Project/src/ttt_ai/agents/hybrid_mcts_agent.py) | **Hoàn thành (Sprint 3–5)**<br>**Hạng 1** giải `3×3, K=3` (`0.637`, `328-872-0`, `0` thua); vượt trội `PureMCTS` & `Policy-only` tại ngân sách `50 sims` (**RQ1**). |

#### Bộ Ấn phẩm Báo cáo, Thuyết trình & Sổ tay Đồ án (`Reports_Project/` & `Jupiter_Notebook/`)

| Ấn phẩm | Vị trí Tệp | Mô tả |
|---|---|---|
| **Báo cáo Đồ án PDF & LaTeX** | [`Report_Extended_TicTacToe.pdf`](Extended_Tic-Tac-Toe_Project/Reports_Project/Report_Extended_TicTacToe.pdf)<br>[`Reports_Project/LaTeX/main.tex`](Extended_Tic-Tac-Toe_Project/Reports_Project/LaTeX/main.tex) | Báo cáo toàn văn 7 chương chuẩn học thuật SGU kèm công thức MDP, phân tích định lượng `RQ1–RQ3`, 5 biểu đồ 300 DPI và đối chiếu Rubric. |
| **Slide Thuyết trình** | [`Slide_Extended_TicTacToe.pptx`](Extended_Tic-Tac-Toe_Project/Reports_Project/Slide_Extended_TicTacToe.pptx)<br>[`Slide_Extended_TicTacToe.pdf`](Extended_Tic-Tac-Toe_Project/Reports_Project/Slide_Extended_TicTacToe.pdf) | Bộ 13 slide thuyết trình chuẩn 16:9 tổng hợp lộ trình thiết kế, kết quả 16,800 ván, phân tích ca lỗi và demo. |
| **Poster Khoa học Khổ A3** | [`Poster_A3_Extended_TicTacToe.pdf`](Extended_Tic-Tac-Toe_Project/Reports_Project/Poster_A3_Extended_TicTacToe.pdf) | Poster báo cáo khoa học khổ A3 ngang dạng PDF vector độ phân giải in ấn cao. |
| **Sổ tay Thực nghiệm Tương tác** | [`Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb`](Extended_Tic-Tac-Toe_Project/Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb) | Notebook 22 cells (6 phần) cho phép chạy thử Engine, đối đầu 7 AI, vẽ Heatmap `Prior` vs `Visit Share` và tái tạo trực tiếp `Fig 1..5`. |

---

## ⚙️ Cài đặt & Môi trường (Prerequisites & Installation)

### Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+, hoặc macOS.
- **Python**: Phiên bản **Python `>= 3.11`** (khuyến nghị `3.11` – `3.12`; đã kiểm thử đầy đủ trên Python `3.12` và `3.14`).
- **Phần cứng**: Tối thiểu 4GB RAM. Có GPU NVIDIA hỗ trợ **CUDA** là điểm cộng khi tái huấn luyện mạng sâu, nhưng toàn bộ quy trình huấn luyện & thực nghiệm chạy mượt mà trên CPU.

### Cài đặt môi trường ảo & Thư viện

#### Bước 1: Clone kho lưu trữ
```bash
git clone https://github.com/ronstudy/Advanced-AI-03.git
cd Advanced-AI-03
```

#### Bước 2: Khởi tạo và kích hoạt môi trường ảo
Trên **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Trên **Linux / macOS (Bash / Zsh)**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Bước 3: Cài đặt package Đồ án Game AI (`ttt-ai`) & thư viện Lab
Cài đặt package đồ án ở chế độ phát triển (`editable`) kèm công cụ kiểm thử (`pytest`, `ruff`) và các gói phục vụ Jupyter Notebooks (Lab 03, Lab 04, Lab 05 & Capstone Notebook):
```bash
python -m pip install --upgrade pip
pip install -e "./Extended_Tic-Tac-Toe_Project[dev]"
pip install pandas numba tabulate jupyter ipykernel
```
*(Tham khảo cấu hình môi trường tự động chuẩn hóa tại [`.agents/environment.json`](.agents/environment.json)).*

---

## 🚀 Hướng dẫn chạy (Usage & Execution)

### 1. Chạy các bài thực hành Lab & Sổ tay Đồ án (Jupyter Notebooks)

Di chuyển vào thư mục tương ứng trước khi khởi động Jupyter để các đường dẫn tệp dữ liệu (`L_maze.txt`) và module nội bộ (`maze_helper.py`, `tictactoe.py`) được nạp chính xác:

```bash
# Lab 03: Local Search
cd Lab/lab03_localsearch_search_TruongChiLong_3123410198
jupyter notebook

# Lab 04: Constraint Satisfaction Problems (CSP)
cd Lab/lab04_csp_TruongChiLong_3123410198
jupyter notebook

# Lab 05: Adversarial Search & Games (Tic-Tac-Toe, Connect 4, Dots and Boxes)
cd Lab/lab05_game_TruongChiLong_3123410198
jupyter notebook

# Đồ án Cuối kỳ: Sổ tay thực nghiệm tương tác 6 phần
cd Extended_Tic-Tac-Toe_Project
jupyter lab Jupiter_Notebook/Extended_Tic_Tac_Toe.ipynb
```

### 2. Chạy kiểm thử tự động (Unit & Integration Tests)

Đồ án Game AI xây dựng hệ thống kiểm thử tự động toàn diện gồm **347 test cases** trên **18 module kiểm thử** (bao phủ Engine, Luật chơi, AST Compliance, 7 Cấp độ AI, Checkpoint SHA-256, Huấn luyện SL/RL, Giải đấu, Ablations, Rich CLI và Visualizer):

```bash
# Chạy toàn bộ 347 tests từ thư mục dự án
cd Extended_Tic-Tac-Toe_Project
pytest -v

# Chạy riêng nhóm kiểm thử Động cơ & Luật chơi O(K)
pytest tests/test_engine.py tests/test_rules.py tests/test_config.py -v

# Chạy nhóm kiểm thử 7 cấp AI & Hợp đồng tác tử (Agent Contract)
pytest tests/test_agents_contract.py tests/test_search_agents.py tests/test_hybrid_mcts_agent.py -v

# Chạy nhóm kiểm thử Giải đấu, Ablations, Rich CLI & Visualizer
pytest tests/test_tournament_runner.py tests/test_ablations.py tests/test_cli_controller.py tests/test_visualizer.py -v
```

### 3. Chạy linter kiểm tra chuẩn mã nguồn

Dự án tuân thủ quy chuẩn **PEP 8** và linter **Ruff** (độ dài dòng $\le 100$, sắp xếp import chuẩn):
```bash
ruff check Extended_Tic-Tac-Toe_Project/src Extended_Tic-Tac-Toe_Project/tests Extended_Tic-Tac-Toe_Project/scripts
```

### 4. Tái lập Thực nghiệm & Khởi chạy Rich CLI Đồ án

Tất cả các bước tái lập thực nghiệm và demo đều chạy bằng **1 lệnh** từ thư mục `Extended_Tic-Tac-Toe_Project/`:

```bash
cd Extended_Tic-Tac-Toe_Project

# 1. Trải nghiệm giao diện Rich CLI Dashboard (Setup Wizard tương tác)
python scripts/play.py

#    Hoặc đấu trực tiếp Người (X) vs HybridMCTS Cấp 7 (O) trên 4x4 K=4 (50 ms/nước)
python scripts/play.py --board 4x4_k4 --mode human_vs_ai --human-player X --agent-o 7 --budget-ms 50

#    Hoặc xem Máy vs Máy từng bước để quan sát Heatmap Prior P(s,a) vs Visit Share N(s,a)/N(s)
python scripts/play.py --board 4x4_k4 --mode ai_vs_ai --agent-x 7 --agent-o 3 --budget-ms 50 --step-by-step

# 2. Sinh dữ liệu chuyên gia & huấn luyện Cấp 5 (FastPolicyNet, StrongPolicyNet, ValueNet)
python scripts/generate_dataset.py --board all --train

# 3. Huấn luyện Cấp 6 đa seed {42, 123, 999} (Tabular Q 3x3 & Double DQN Self-Play 4x4)
python scripts/train_all.py --mode all --board all

# 4. Chạy bộ 4 giải đấu Paired Round-Robin chính thức (16,800 ván) hoặc chỉ kiểm chứng log có sẵn
python scripts/run_tournament.py --suite
python scripts/run_tournament.py --suite --verify-only

# 5. Chạy 4 kịch bản thực nghiệm Ablation khoa học (8,756 ván: AB Pruning, RQ1, RQ2, RQ3)
python scripts/run_ablations.py --ablation all --mode full --seed 42

# 6. Kết xuất tự động 5 biểu đồ báo cáo (300 DPI) & toàn bộ bảng tổng hợp từ raw logs
python scripts/plot_results.py --figures all --dpi 300
```

---

## 📊 Kết quả & Đánh giá (Experiments & Evaluation)

### 1. Thực nghiệm Tìm kiếm cục bộ (Lab 03)
- **Mê cung chữ L (Local Optima)**: Thuật toán *Simple Hill Climbing* xuất phát từ $S(9, 5)$ ($h=14$) luôn rơi vào bẫy cực trị cục bộ tại tọa độ $(5, 9)$ ($h=6$) và không thể tiến tới đích $G(2, 12)$ bất kể thứ tự duyệt hướng (`NESW`, `ESWN`,...).
- **So sánh các biến thể trên $n$-Queens ($n=8$)**:
  - *Steepest-Ascent* và *Stochastic HC* đơn lẻ đạt tỷ lệ thành công **12.0% – 14.0%**; riêng *Efficient Stochastic HC (First-choice)* giảm hơn **55% thời gian chạy** so với *Steepest-Ascent*.
  - *Random Restart Hill Climbing* đạt tỷ lệ thành công **100%** (trung bình **6.8 lần restart**).
  - *Simulated Annealing* với lịch Exponential ($T_0=4.0, \alpha=0.995$) đạt **98.0%**. Trong 4 toán tử di chuyển, `move_adaptive` hội tụ nhanh và ổn định nhất.
- **Tăng tốc Numba JIT**: Trên quy mô lớn ($n=450$), bản Numba `@njit` tính xung đột nhanh hơn phiên bản Python tuyến tính $O(n)$ gấp **90.3 lần** và vượt trội bản Naive $O(n^2)$ hàng nghìn lần.

### 2. Thực nghiệm Bài toán thỏa mãn ràng buộc (Lab 04)
- **Tô màu đồ thị phẳng (Graph Coloring)**:
  - Với 4 màu ($k=4, n=10$ đỉnh): Backtracking cơ bản, MRV/LCV, Forward Checking và Min-Conflicts đều giải thành công **100%** (trung bình 8–11 nút mở rộng).
  - Với 3 màu ($k=3$): Min-Conflicts đạt 0% do kẹt cực trị cục bộ khi sắc số $\chi(G) > 3$.
- **Độ mở rộng (Scalability Analysis)**: Độ dốc Log-Log của *Simulated Annealing* ($1.06$) nhỏ hơn rõ rệt so với *Efficient Stochastic HC* ($1.92$), khẳng định khả năng mở rộng tốt hơn khi $n$ tăng.

### 3. Thực hành Tìm kiếm đối kháng & Trò chơi (Lab 05)
- Hoàn thiện chuỗi 7 notebook giải Tic-Tac-Toe từ AND-OR Tree Search, Minimax/Alpha-Beta toàn cây, Heuristic Alpha-Beta giới hạn độ sâu đến Pure Monte Carlo và MCTS UCB1, tạo tiền đề thuật toán trực tiếp cho đồ án cuối kỳ.
- Áp dụng thành công tìm kiếm đối kháng, thiết kế heuristic và kỹ thuật profiling / quản lý trạng thái agent cho 3 bài toán mở rộng: **Connect 4**, **"Mean" Connect 4**, và **Dots and Boxes**.

### 4. Kết quả Chính thức Đồ án Cuối kỳ Extended Tic-Tac-Toe (5/5 Sprint)

Toàn bộ **5/5 Sprint** đã hoàn thành, đạt **347/347 tests pass** và **Ruff PEP 8 clean** (`All checks passed!`). Mọi số liệu dưới đây được tổng hợp tự động trực tiếp từ **25,640 ván** trong `logs/*.jsonl` (chi tiết đầy đủ tại [`summary_tables.md`](Extended_Tic-Tac-Toe_Project/reports/summary_tables.md), [`ablation_summary.md`](Extended_Tic-Tac-Toe_Project/reports/ablation_summary.md) và [`sample_failure_cases.md`](Extended_Tic-Tac-Toe_Project/reports/sample_failure_cases.md)).

#### a) Tổng hợp 4 Giải đấu Paired Round-Robin Chính thức (16,800 ván)

| Giải đấu | Bàn cờ | Ngân sách | Số ván | Hạng 1 (Top Agent) | Top Score `[95% CI]` | Seat $\Delta\text{Score}$ `[95% CI]` |
|:---|:---:|:---|---:|:---|:---:|:---:|
| `tournament_3x3_k3_default` | `3×3, K=3` | `default` | 4,200 | **L7_HybridMCTS** (`328-872-0`) | `0.637 [0.624, 0.650]` | `+0.141 [+0.124, +0.159]` |
| `tournament_4x4_k4_default` | `4×4, K=4` | `default` | 4,200 | **L4_MCTS** (`282-911-7`) | `0.615 [0.603, 0.627]` | `+0.015 [-0.000, +0.031]` |
| `tournament_4x4_k4_fixed_10ms` | `4×4, K=4` | `fixed_10ms` | 4,200 | **L4_MCTS** (`275-877-48`) | `0.595 [0.581, 0.608]` | `+0.020 [+0.005, +0.038]` |
| `tournament_4x4_k4_fixed_50ms` | `4×4, K=4` | `fixed_50ms` | 4,200 | **L4_MCTS** (`273-922-5`) | `0.612 [0.600, 0.625]` | `+0.015 [-0.000, +0.031]` |

#### b) Kết luận Định lượng từ 4 Thí nghiệm Ablation (8,756 ván — Trả lời `RQ1, RQ2, RQ3`)

1. **Hiệu quả Cắt tỉa Alpha-Beta + Move Ordering (`ABLATION_1`, 316 ván trên `4×4, K=4`)**:
   - Bảo toàn chính xác tuyệt đối giá trị gốc (`Exact Root Value = YES`) ở mọi độ sâu `d=2, 3, 4`.
   - Tại `d=4`, số nút duyệt trung bình giảm từ **23,171.0** (Minimax thuần) $\rightarrow$ **2,035.5** (Alpha-Beta không sắp xếp, `-91.22%`) $\rightarrow$ **1,225.0** (Alpha-Beta + Move Ordering, **giảm `-94.71%`**).
2. **RQ1 — Hybrid MCTS vs Pure MCTS & Policy-only (`ABLATION_2`, 2,760 ván trên `4×4, K=4`)**:
   - Ở ngân sách mô phỏng thấp (`50 sims`), `HybridRollout_sim50` (`0.583 [0.542, 0.625]`) và `HybridMixed_sim50` (`0.575 [0.533, 0.621]`) vượt trội cả `PureMCTS_sim50` (`0.529 [0.471, 0.588]`) lẫn `Policy_Strong` (`0.525`) và `Value_Greedy` (`0.296`).
   - Ở giới hạn thời gian CPU chặt (`10 ms` / `50 ms`), chi phí forward PyTorch làm giảm số mô phỏng mỗi nước (`145.9` vs `190.7 sims` ở `10 ms`), nhưng bộ đệm **LRU Cache** đạt tỷ lệ trúng **`60.7%–69.0%`**. Trên `3×3, K=3`, `L7_HybridMCTS` xếp **hạng 1 toàn giải (`0` thua)**.
3. **RQ2 — Lợi thế Người đi trước theo Luật thắng `K` (`ABLATION_3`, 2,080 ván)**:
   - `4×4, K=3`: Lợi thế đi trước bùng nổ với `X Win = 76.5%`, `Draw = 0.0%`, $\Delta\text{Score} = +0.530\;[+0.445, +0.610]$.
   - `3×3, K=3`: `X Win = 23.8%`, `Draw = 63.6%`, $\Delta\text{Score} = +0.112\;[+0.073, +0.152]$.
   - `4×4, K=4`: Mở rộng không gian phòng thủ kéo tỷ lệ hòa lên `74.1%` và đưa $\Delta\text{Score}$ về gần cân bằng $+0.024\;[-0.011, +0.058]$.
4. **RQ3 — Tổng quát hóa & Độ ổn định Đa Seed: RL Self-Play vs Supervised (`ABLATION_4`, 3,600 ván)**:
   - Trên `4×4, K=4`, `DQNAgent` (3 seeds `{42, 123, 999}`) đạt **`0.586 ± 0.027`** tổng thể và **`0.554 ± 0.016`** trước đối thủ chưa từng gặp lúc huấn luyện (*Unseen `MCTS`*), vượt xa `SL_StrongPolicy` (`0.497` tổng thể, `0.344` trước `MCTS`) và `SL_ValueGreedy` (`0.307`).
   - Trên `3×3, K=3`, `TabularQAgent` (3 seeds) đạt **`0.659 ± 0.005`** (`0.619 ± 0.023` trước `MCTS`) so với `0.469–0.485` của nhóm Supervised.

#### c) Bộ 5 Biểu đồ Khoa học 300 DPI (`reports/figures/`)
- **Fig 1:** [`fig1_head_to_head_matrix.png`](Extended_Tic-Tac-Toe_Project/reports/figures/fig1_head_to_head_matrix.png) — Ma trận đối đầu kép (`As X` vs `As O`) 7 cấp AI.
- **Fig 2:** [`fig2_latency_and_nodes.png`](Extended_Tic-Tac-Toe_Project/reports/figures/fig2_latency_and_nodes.png) — Độ trễ `mean_ms` (thang log), `p95_ms` và chi phí nút/mô phỏng.
- **Fig 3:** [`fig3_ablation_rq1_and_alphabeta.png`](Extended_Tic-Tac-Toe_Project/reports/figures/fig3_ablation_rq1_and_alphabeta.png) — Kết quả **RQ1** (Hybrid vs Pure MCTS / Policy) & Alpha-Beta Pruning.
- **Fig 4:** [`fig4_ablation_rq2_seat_advantage.png`](Extended_Tic-Tac-Toe_Project/reports/figures/fig4_ablation_rq2_seat_advantage.png) — Kết quả **RQ2** (Dịch chuyển tỷ lệ thắng/hòa và $\Delta\text{Score}$ kèm 95% CI).
- **Fig 5:** [`fig5_ablation_rq3_learning_and_generalization.png`](Extended_Tic-Tac-Toe_Project/reports/figures/fig5_ablation_rq3_learning_and_generalization.png) — Kết quả **RQ3** (Learning curves đa seed & Unseen Opponents).

---

## 📚 Nguồn Tham khảo & Kế thừa (References & Inheritance)

### 1. Tài liệu môn học & Giáo trình kinh điển
- [`Tic tac toe Game.pdf`](Extended_Tic-Tac-Toe_Project/References/Tic%20tac%20toe%20Game.pdf) — Slide định hướng trò chơi Tic-Tac-Toe mở rộng, kiến trúc mạng đối chứng và phương pháp tìm kiếm trong không gian trạng thái.
- **Stuart Russell & Peter Norvig**, *Artificial Intelligence: A Modern Approach (4th Edition)*, Pearson, 2020 — [`Artificial Intelligence A Modern Approach 4th.pdf`](Extended_Tic-Tac-Toe_Project/References/Artificial%20Intelligence%20A%20Modern%20Approach%204th.pdf).
- **Richard S. Sutton & Andrew G. Barto**, *Reinforcement Learning: An Introduction (2nd Edition)*, The MIT Press, 2018 — [`SuttonBartoIPRLBook2ndEd.pdf`](Extended_Tic-Tac-Toe_Project/References/SuttonBartoIPRLBook2ndEd.pdf).
- **Mark Liu**, *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*, 2024 — [`2024 - Liu - AlphaGo Simplified...pdf`](Extended_Tic-Tac-Toe_Project/References/2024%20-%20Liu%20-%20AlphaGo%20Simplified%20Rule-Based%20AI%20and%20Deep%20Learning%20in%20Everyday%20Games.pdf) và mã nguồn đối chiếu tại [`AlphaGoSimplified-main/`](Extended_Tic-Tac-Toe_Project/References/AlphaGoSimplified-main/).

### 2. Mã nguồn tham khảo & Khung thực hành
- Khung bài tập thực hành **Lab 03, Lab 04, Lab 05** ([`maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py), [`tictactoe.py`](Lab/lab05_game_TruongChiLong_3123410198/tictactoe.py)) tham khảo từ học liệu của **Michael Hahsler** (Khoa Khoa học Máy tính, SMU). Toàn bộ phần giải thuật mở rộng, tối ưu hóa Numba, bài tập Connect 4 / Dots and Boxes, đối sánh thực nghiệm và báo cáo do sinh viên thực hiện.
- [`Reversi-mcts-master/`](Extended_Tic-Tac-Toe_Project/References/Reversi-mcts-master/) — Dự án mã nguồn C#/.NET tham khảo cấu trúc 4 pha MCTS UCT trong trò chơi đối kháng bàn cờ. Toàn bộ mã nguồn Đồ án `Extended_Tic-Tac-Toe_Project` được thiết kế và cài đặt mới 100% bằng Python/PyTorch (xem bảng đối chiếu minh bạch học thuật tại [`Extended_Tic-Tac-Toe_Project/README.md`](Extended_Tic-Tac-Toe_Project/README.md#8-minh-bạch-học-thuật--kế-thừa)).
