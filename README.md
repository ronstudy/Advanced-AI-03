# Advanced-AI-03 — Trí Tuệ Nhân Tạo Nâng Cao (Advanced AI Lab & Projects)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Tests-216%20passed-brightgreen?logo=pytest&logoColor=white" alt="Pytest Status" />
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-black?logo=ruff&logoColor=white" alt="Code Style" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</p>

> Kho lưu trữ mã nguồn, bài tập lý thuyết, báo cáo thực hành và đồ án Game AI môn **Trí Tuệ Nhân Tạo Nâng Cao**. Dự án bao trùm các lĩnh vực nền tảng và nâng cao của AI cổ điển và hiện đại: từ các thuật toán Tìm kiếm (Uninformed/Informed Search, Local Search), Bài toán thỏa mãn ràng buộc (CSP), đến Lý thuyết trò chơi đối kháng (Minimax, Alpha-Beta Pruning, MCTS UCT), Mạng nơ-ron học có giám sát và Học tăng cường (Reinforcement Learning).

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
    - [2. Thực hành Lab (Practical Labs)](#2-thực-hành-lab-practical-labs)
    - [3. Đồ án Cuối kỳ: Game AI Đa Cấp Độ (Capstone Project)](#3-đồ-án-cuối-kỳ-game-ai-đa-cấp-độ-capstone-project)
  - [⚙️ Cài đặt & Môi trường (Prerequisites & Installation)](#️-cài-đặt--môi-trường-prerequisites--installation)
    - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
    - [Cài đặt môi trường ảo & Thư viện](#cài-đặt-môi-trường-ảo--thư-viện)
  - [🚀 Hướng dẫn chạy (Usage & Execution)](#-hướng-dẫn-chạy-usage--execution)
    - [1. Chạy các bài thực hành Lab (Jupyter Notebooks)](#1-chạy-các-bài-thực-hành-lab-jupyter-notebooks)
    - [2. Chạy kiểm thử tự động (Unit Tests)](#2-chạy-kiểm-thử-tự-động-unit-tests)
    - [3. Chạy linter kiểm tra chuẩn mã nguồn](#3-chạy-linter-kiểm-tra-chuẩn-mã-nguồn)
    - [4. Khởi chạy Game AI & Scripts Đồ án](#4-khởi-chạy-game-ai--scripts-đồ-án)
  - [📊 Kết quả & Đánh giá (Experiments & Evaluation)](#-kết-quả--đánh-giá-experiments--evaluation)
    - [1. Thực nghiệm Tìm kiếm cục bộ (Lab 03)](#1-thực-nghiệm-tìm-kiếm-cục-bộ-lab-03)
    - [2. Thực nghiệm Bài toán thỏa mãn ràng buộc (Lab 04)](#2-thực-nghiệm-bài-toán-thỏa-mãn-ràng-buộc-lab-04)
    - [3. Kiểm thử & Sanity Benchmark Đồ án Game AI (Sprint 1 & Sprint 2)](#3-kiểm-thử--sanity-benchmark-đồ-án-game-ai-sprint-1--sprint-2)
  - [📚 Nguồn Tham khảo & Kế thừa (References & Inheritance)](#-nguồn-tham-khảo--kế-thừa-references--inheritance)
  - [👤 Tác giả & Giấy phép (Author & License)](#-tác-giả--giấy-phép-author--license)

---

## 🗂️ Cấu trúc Repository (Project Structure)

```text
Advanced-AI-03/
├── README.md                                             # Tài liệu tổng quan toàn bộ repository
├── Ex/                                                   # Bài tập lý thuyết (Bản nộp PDF)
│   ├── ex01_AISearch_TruongChiLong_3123410198.pdf        # Lý thuyết Tìm kiếm & Heuristic trong AI
│   ├── ex02_CSPGame_TruongChiLong_3123410198.pdf         # Lý thuyết CSP, Minimax & Alpha-Beta
│   └── ex03_AIGame_TruongChiLong_3123410198.pdf          # Lý thuyết Game nâng cao, Expectimax & MDP
├── Lab/                                                  # Bài thực hành lập trình, notebooks & báo cáo
│   ├── lab03_localsearch_search_TruongChiLong_3123410198/# Lab 03: Thuật toán tìm kiếm cục bộ (Local Search)
│   │   ├── README.md                                     # Mục lục & hướng dẫn bài Lab 03
│   │   ├── BAOCAO_lab03_localsearch_search.pdf           # Báo cáo chi tiết thực nghiệm Lab 03
│   │   ├── L_maze.txt                                    # Dữ liệu ma trận mê cung chữ L
│   │   ├── maze_helper.py                                # Tiện ích nạp bản đồ, vẽ mê cung & heuristic
│   │   ├── Maze_Example_LocalSearch.ipynb                # Simple Hill Climbing trên mê cung chữ L
│   │   ├── n_queens.ipynb                                # n-Queens: HC variants, SA, GA, 4 operators
│   │   ├── n_queens_fast_conflict_calculation_with_numba.ipynb # Tối ưu tính xung đột n-Queens bằng Numba JIT
│   │   ├── Simulated_Annealing_Schedules.ipynb           # Khảo sát 4 lịch giảm nhiệt độ Simulated Annealing
│   │   └── traveling_salesman_problem.ipynb              # TSP: Local Search (2-swap vs 2-opt), GA
│   └── lab04_csp_TruongChiLong_3123410198/               # Lab 04: Bài toán thỏa mãn ràng buộc (CSP)
│       ├── README.md                                     # Mục lục & hướng dẫn bài Lab 04
│       ├── lab04_csp_TruongChiLong_3123410198.pdf        # Báo cáo thực nghiệm chi tiết Lab 04
│       ├── CSP_graph_coloring_example.ipynb              # CSP: Tô màu đồ thị (Backtracking, MRV, LCV, FC, Min-Conflicts)
│       ├── n_queens.ipynb                                # CSP / Local Search n-Queens & hội tụ Log-Log
│       ├── Graph Coloring.png                            # Biểu đồ trực quan hóa đồ thị đã tô màu hợp lệ
│       ├── Queen.png                                     # Trực quan hóa nghiệm n-Queens
│       ├── Algorithm Convergence.png                     # Biểu đồ tốc độ hội tụ qua các bước lặp
│       └── Runtime Scalability.png                       # Đồ thị độ phức tạp thời gian theo kích thước bài toán
├── Midterm/                                              # Thư mục lưu trữ tài liệu / đề cương ôn tập giữa kỳ
└── Advanced_AI_Tic-Tac-Toe_Project/                      # Đồ án Cuối kỳ: Game AI Đa Cấp Độ (M×N, K)
    ├── pyproject.toml                                    # Cấu hình cài đặt package ttt-ai, pytest & ruff
    ├── Planning/                                         # Bộ 6 tài liệu thiết kế, quy chuẩn & quản lý dự án
    │   ├── PRD.md                                        # Yêu cầu sản phẩm, mô hình MDP, 3 câu hỏi nghiên cứu RQ
    │   ├── RULES.md                                      # Quy định trung thực học thuật, đo đạc & code standard
    │   ├── ARCHITECTURE.md                               # Kiến trúc phân tầng, API chuẩn, chi tiết 7 cấp AI
    │   ├── DESIGN.md                                     # Thiết kế giao diện Rich CLI & 5 bộ biểu đồ báo cáo
    │   ├── TASKS.md                                      # Phân rã công việc 5 Sprint & điều kiện nghiệm thu DoD
    │   └── PROGRESS.md                                   # Nhật ký tiến độ thực tế, registry thực nghiệm & lỗi
    ├── References/                                       # Đề cương, tài liệu & mã nguồn mở đối chiếu
    │   ├── TTNTNC_De_tai_cuoi_ky_Game_AI.pdf             # Đề tài cuối kỳ Game AI chính thức từ GVHD
    │   ├── Tic tac toe Game.pdf                          # Tài liệu định hướng trò chơi Tic-Tac-Toe mở rộng
    │   ├── AlphaGoSimplified-main/                       # Mã nguồn đối chiếu mô hình AlphaGo đơn giản hóa
    │   └── Reversi-mcts-master/                          # Mã nguồn tham khảo thuật toán MCTS & BTMM
    ├── configs/                                          # Tệp cấu hình YAML cho trò chơi & thực nghiệm
    │   ├── game_3x3_k3.yaml                              # Cấu hình bàn cờ tiêu chuẩn 3×3, K=3
    │   ├── game_4x4_k3.yaml                              # Biến thể bàn cờ 4×4, K=3
    │   ├── game_4x4_k4.yaml                              # Bàn cờ chuẩn chính thức 4×4, K=4
    │   ├── game_5x5_k4.yaml                              # Bàn cờ mở rộng 5×5, K=4
    │   ├── tournament.yaml                               # Cấu hình giải đấu Paired Round-Robin
    │   ├── train_supervised.yaml                         # Cấu hình huấn luyện mạng Policy/Value
    │   └── train_rl.yaml                                 # Cấu hình huấn luyện tự chơi Q-Learning/DQN
    ├── src/ttt_ai/                                       # Mã nguồn lõi package ttt_ai
    │   ├── __init__.py                                   # Khởi tạo package ttt_ai
    │   ├── config.py                                     # Nạp và kiểm định cấu hình, hàm seed_everything()
    │   ├── engine/                                       # Động cơ trò chơi (Core Game Engine)
    │   │   ├── exceptions.py                             # Hệ thống ngoại lệ tùy biến cho Game Engine
    │   │   ├── rules.py                                  # Thuật toán kiểm tra thắng O(K), canonical board, weights
    │   │   └── game.py                                   # Lớp ExtendedTicTacToe (7+1 API chuẩn, render ASCII/Rich)
    │   ├── agents/                                       # 7 cấp độ tác tử AI đối kháng
    │   │   ├── base.py                                   # BaseAgent, AgentDecision & SearchTimeoutError
    │   │   ├── random_agent.py                           # Cấp 1: Random baseline
    │   │   ├── rule_based_agent.py                       # Cấp 2: Rule-Based (6 tầng ưu tiên Newell & Simon)
    │   │   ├── minimax_agent.py                          # Cấp 3: Minimax / Alpha-Beta + Iterative Deepening
    │   │   ├── mcts_agent.py                             # Cấp 4: Pure MCTS UCT 4 pha
    │   │   ├── nn_models.py                              # Mạng Policy/Value (FastPolicyNet, StrongPolicyNet, ValueNet)
    │   │   ├── policy_agent.py                           # Cấp 5: Supervised Policy / Value Greedy Agent
    │   │   ├── rl_agent.py                               # Cấp 6: Tabular Q-Learning & Self-Play DQN Agent
    │   │   └── hybrid_mcts_agent.py                      # Cấp 7: Hybrid MCTS PUCT guided by Neural Networks
    │   ├── training/                                     # Module huấn luyện & sinh dữ liệu
    │   ├── benchmark/                                    # Module đo đạc, giải đấu & ablations
    │   └── ui/                                           # Giao diện Rich Terminal CLI
    ├── tests/                                            # Hệ thống kiểm thử tự động (216 passed tests)
    │   ├── test_config.py                                # Kiểm thử nạp YAML, schema & seed
    │   ├── test_rules.py                                 # Kiểm thử O(K) win check, sliding windows & canonical
    │   ├── test_engine.py                                # Kiểm thử 7+1 Game API, immutability, copy/undo
    │   ├── test_base_agent.py                            # Kiểm thử interface BaseAgent & timeout guard
    │   ├── test_level1_level2_agents.py                  # Kiểm thử hành vi RandomAgent & 6 tầng luật RuleBased
    │   ├── test_minimax_agent.py                         # Kiểm thử Minimax, AlphaBeta pruning & heuristic
    │   ├── test_mcts_agent.py                            # Kiểm thử 4 pha UCT, rollouts & backpropagation
    │   ├── test_agents_contract.py                       # Kiểm thử tính nhất quán hợp đồng của tất cả Agents
    │   └── test_search_agents.py                         # Kiểm thử benchmark và cắt tỉa nút của Search Agents
    ├── terminal_test/                                    # Ảnh chụp minh chứng kết quả kiểm thử terminal Sprint 1–2
    ├── checkpoints/                                      # Thư mục lưu trữ trọng số mô hình PyTorch (*.pt)
    ├── logs/                                             # Thư mục xuất dữ liệu log thực nghiệm JSON Lines (*.jsonl)
    ├── reports/                                          # Thư mục xuất biểu đồ và báo cáo markdown
    └── scripts/                                          # Các script CLI vận hành hệ thống
        ├── play.py                                       # Script khởi chạy giao diện đấu cờ Terminal
        ├── train_all.py                                  # Script tự động hóa toàn bộ quy trình huấn luyện
        ├── run_tournament.py                             # Script khởi chạy giải đấu vòng tròn tính điểm
        ├── run_ablations.py                              # Script chạy 4 kịch bản thực nghiệm cô lập (Ablations)
        └── plot_results.py                               # Script vẽ biểu đồ và tổng hợp số liệu báo cáo
```

---

## 🔬 Nội dung & Thuật toán triển khai (Core Topics & Algorithms)

### 1. Bài tập Lý thuyết (Theory Exercises)

| STT | Bài toán / Chủ đề | Thuật toán / Kỹ thuật chính | File / Tài liệu chính | Trạng thái / Kết quả |
|:---:|---|---|---|:---:|
| **01** | Tìm kiếm & Heuristic trong Không gian trạng thái | BFS, DFS, IDS, UCS, Greedy Best-First Search, A\*; Phân tích tính *admissible* và *consistent* của heuristic (khoảng cách Manhattan, vector vận tốc); Mô hình hóa bài toán Tháp Hà Nội ($3^n$ trạng thái). | [`Ex/ex01_AISearch_TruongChiLong_3123410198.pdf`](Ex/ex01_AISearch_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** (100% lời giải chuẩn xác) |
| **02** | Bài toán Thỏa mãn Ràng buộc & Game AI cơ bản | Nhận diện Local Optima và Shoulders trong Local Search; Cây Minimax và tỉa nhánh Alpha-Beta; Expectiminimax và ngưỡng cắt tỉa xác suất; Thuật toán Forward Checking, Arc Consistency (AC-3), Heuristic MRV và LCV (Ma phương 3×3, Lập lịch đường băng, Tô màu đồ thị $K_3$). | [`Ex/ex02_CSPGame_TruongChiLong_3123410198.pdf`](Ex/ex02_CSPGame_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** (100% lời giải chuẩn xác) |
| **03** | Tìm kiếm nâng cao, Game đối kháng & Quá trình Quyết định Markov | Đánh giá cây Minimax/Alpha-Beta nhiều tầng; Cắt tỉa nhánh ngẫu nhiên trong Expectimax; So sánh chính sách Minimax vs Expectimax vs Value Iteration (MDP); Lan truyền ràng buộc AC-3 theo thứ tự topo; Trò chơi nhiều mục tiêu và hợp tác Pacman–Casper; Cập nhật động đường đi A\* khi thay đổi trọng số cạnh. | [`Ex/ex03_AIGame_TruongChiLong_3123410198.pdf`](Ex/ex03_AIGame_TruongChiLong_3123410198.pdf) | **Đã hoàn thành** (100% lời giải chuẩn xác) |

---

### 2. Thực hành Lab (Practical Labs)

| STT | Bài toán / Chủ đề | Thuật toán / Kỹ thuật chính | File / Notebook chính | Trạng thái / Kết quả |
|:---:|---|---|---|:---:|
| **04** | Tìm kiếm đường đi trong Mê cung chữ L (Lab 03) | Simple Hill Climbing với hàm Heuristic Manhattan $h(n) = \|r - r_G\| + \|c - c_G\|$; Khảo sát các thứ tự ưu tiên hướng di chuyển (`NESW`, `ESWN`, `SWNE`, `WNES`) và ngẫu nhiên hóa. | [`Lab/.../Maze_Example_LocalSearch.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Maze_Example_LocalSearch.ipynb)<br>[`Lab/.../maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) | **Đã hoàn thành**<br>Chứng minh thuật toán kẹt tại cực trị cục bộ $(5, 9)$ với $h=6$ do tường chắn chữ L. |
| **05** | Bài toán $n$-Queens bằng Tìm kiếm cục bộ (Lab 03) | Cài đặt & so sánh 5 thuật toán: Steepest-Ascent HC, Inefficient Stochastic HC, Efficient (First-choice) Stochastic HC, Random Restart HC, Simulated Annealing. Cài đặt 4 toán tử di chuyển nâng cao (`single_step`, `col_swap`, `dual_queen`, `adaptive`) và Genetic Algorithm. | [`Lab/.../n_queens.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens.ipynb)<br>[`Lab/.../BAOCAO_lab03_localsearch_search.pdf`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/BAOCAO_lab03_localsearch_search.pdf) | **Đã hoàn thành**<br>Random Restart HC đạt 100% thành công (TB ~6.8 lần restart); SA đạt 98% ($n=8$). Toán tử `move_adaptive` hội tụ nhanh nhất. |
| **06** | Khảo sát Lịch giảm nhiệt độ trong Simulated Annealing (Lab 03) | Khảo sát 4 quy luật làm lạnh $T(t)$: *Exponential* ($T = T_0 \alpha^t$), *Linear* ($T = T_0 - \beta t$), *Fast* ($T = \frac{T_0}{1 + t}$), và *Logarithmic* ($T = \frac{T_0}{1 + \ln(1+t)}$). Phân tích cán cân Exploration vs Exploitation. | [`Lab/.../Simulated_Annealing_Schedules.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Simulated_Annealing_Schedules.ipynb) | **Đã hoàn thành**<br>Quy luật Exponential với $\alpha \in [0.99, 0.995]$ đạt hiệu quả cân bằng tối ưu nhất. |
| **07** | Tối ưu hóa tính toán xung đột $n$-Queens bằng Numba JIT (Lab 03) | So sánh độ phức tạp và hiệu năng: Hàm Naive $O(n^2)$, Hàm Linear $O(n)$ dùng mảng tần suất và tổ hợp $\binom{k}{2}$, và Hàm JIT Compiled Numba (`@njit`). | [`Lab/.../n_queens_fast_conflict_calculation_with_numba.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens_fast_conflict_calculation_with_numba.ipynb) | **Đã hoàn thành**<br>Trên $n=450$, bản Numba JIT nhanh hơn bản Python thuần $O(n)$ khoảng **90.3×** và nhanh hơn Naive $O(n^2)$ hàng nghìn lần. |
| **08** | Bài toán Người du lịch TSP (Lab 03) | Áp dụng Local Search giải TSP Euclid ($n \in \{10, 20, 30, 50\}$). So sánh toán tử hoán vị `2-swap` với toán tử đảo đoạn `2-opt`; Phần mở rộng Genetic Algorithm với Order Crossover (OX). | [`Lab/.../traveling_salesman_problem.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/traveling_salesman_problem.ipynb) | **Đã hoàn thành**<br>Toán tử `2-opt` tháo gỡ triệt để các giao cắt cạnh trên mặt phẳng, cho chiều dài lộ trình ngắn hơn rõ rệt so với `2-swap`. |
| **09** | Bài toán Tô màu đồ thị CSP (Lab 04) | Biểu diễn CSP trên đồ thị Delaunay tam giác hóa phẳng (`scipy.spatial.Delaunay`). Cài đặt Backtracking Search cơ bản kết hợp kỹ thuật heuristic: **MRV** (Minimum Remaining Values), **LCV** (Least Constraining Value), **Forward Checking (FC)** và **Min-Conflicts Local Search**. | [`Lab/.../CSP_graph_coloring_example.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/CSP_graph_coloring_example.ipynb)<br>[`Lab/.../Graph Coloring.png`](Lab/lab04_csp_TruongChiLong_3123410198/Graph%20Coloring.png) | **Đã hoàn thành**<br>Với 4 màu: 100% thuật toán giải quyết trọn vẹn (TB 8-11 nút mở rộng). Với 3 màu: Min-Conflicts kẹt 0% do sắc số $\chi(G) > 3$. |
| **10** | Mô hình hóa $n$-Queens dưới góc nhìn CSP (Lab 04) | Phân tích $n$-Queens như bài toán ràng buộc với biến $X_i \in \{1..n\}$. So sánh Backtracking CSP với Local Search; Khảo sát độ phức tạp thời gian theo thang đo Log-Log (độ dốc runtime slope). | [`Lab/.../n_queens.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/n_queens.ipynb)<br>[`Lab/.../lab04_csp_TruongChiLong_3123410198.pdf`](Lab/lab04_csp_TruongChiLong_3123410198/lab04_csp_TruongChiLong_3123410198.pdf) | **Đã hoàn thành**<br>Xác định hệ số góc Log-Log: Efficient Stochastic HC ($1.92$) vs Simulated Annealing ($1.06$), chứng minh SA mở rộng tốt hơn khi $n$ lớn. |

---

### 3. Đồ án Cuối kỳ: Game AI Đa Cấp Độ (Capstone Project)

Dự án phát triển hệ thống tác tử đối kháng 7 cấp độ cho trò chơi **Extended Tic-Tac-Toe ($M \times N, K$)** trên các biến thể bàn cờ: Chuẩn chính ($4 \times 4, K=4$), Biến thể ($4 \times 4, K=3$), Đối chứng ($3 \times 3, K=3$) và Mở rộng ($5 \times 5, K=4$).

| Cấp độ | Tác tử AI | Thuật toán / Nguyên lý hoạt động | Module triển khai chính | Trạng thái kiểm thử / Thực nghiệm |
|:---:|---|---|---|:---:|
| **L0** | **Core Engine & Rules** | Động cơ trò chơi chuẩn PEP 8; Kiểm tra thắng gia tăng $O(K)$; Biểu diễn Canonical Board (góc nhìn người chơi hiện tại); Bất biến hóa hành động; Tính đối xứng 8 phép biến đổi D4. | [`src/ttt_ai/engine/`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/engine/) | **Đã hoàn thành (Sprint 1)**<br>Passing 80/80 unit tests (`test_engine.py`, `test_rules.py`). |
| **L1** | **RandomAgent** | Lựa chọn hành động ngẫu nhiên đồng đều trong không gian nước đi hợp lệ (`legal_actions`). Tác tử đối chứng baseline. | [`src/ttt_ai/agents/random_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/random_agent.py) | **Đã hoàn thành (Sprint 2)**<br>Hoàn thành hợp đồng Agent API; Tốc độ ra quyết định $\approx 0.02$ ms/nước. |
| **L2** | **RuleBasedAgent** | Hệ thống luật tri thức chuyên gia dựa trên Newell & Simon tổng quát hóa 6 tầng ưu tiên: (1) Thắng ngay, (2) Chặn thua ngay, (3) Tạo nước đôi (Fork), (4) Chặn nước đôi, (5) Chiếm trung tâm, (6) Nước đi tiềm năng tối ưu. | [`src/ttt_ai/agents/rule_based_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/rule_based_agent.py) | **Đã hoàn thành (Sprint 2)**<br>Bất bại vét cạn 100% trên $3 \times 3, K=3$; Thắng áp đảo RandomAgent với Score $0.935$ trên $4 \times 4, K=4$ ($0.06$ ms/nước). |
| **L3** | **MinimaxAgent / AlphaBetaAgent** | Tìm kiếm đối kháng cạn kèm **Alpha-Beta Pruning**; Heuristic đánh giá cửa sổ trượt $K$ ô; Sắp xếp nước đi (**Move Ordering**) theo trọng số ô; Lặp sâu dần (**Iterative Deepening**) kiểm soát ngân sách thời gian thực. | [`src/ttt_ai/agents/minimax_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/minimax_agent.py) | **Đã hoàn thành (Sprint 2)**<br>Alpha-Beta + Move Ordering **cắt giảm 95.9% số nút duyệt** trên bàn $4 \times 4$ (từ $47,297$ xuống $1,941$ nút). |
| **L4** | **MCTSAgent (UCT)** | Monte Carlo Tree Search 4 pha chuẩn mực: **Selection** (công thức UCB1), **Expansion**, **Simulation** (Rollout ngẫu nhiên có trọng số), **Backpropagation**; Hỗ trợ nhiễu Dirichlet tại nút gốc để khám phá. | [`src/ttt_ai/agents/mcts_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/mcts_agent.py) | **Đã hoàn thành (Sprint 2)**<br>Đạt điểm số $1.000$ (40-0-0) trước RandomAgent ($N=200$ sims); Hòa tuyệt đối $0.500$ (0 thua) trước RuleBased và AlphaBeta. |
| **L5** | **PolicyNetworkAgent / ValueGreedyAgent** | Mạng nơ-ron tích chập (CNN) và MLP học có giám sát: `FastPolicyNet`, `StrongPolicyNet` và `ValueNet` có Action Masking; Dự đoán phân phối xác suất nước đi và giá trị trạng thái. | [`src/ttt_ai/agents/nn_models.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/nn_models.py)<br>[`src/ttt_ai/agents/policy_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/policy_agent.py) | **Khung đã thiết kế (Sprint 3)**<br>Chuẩn bị tập dữ liệu game-level split 70/15/15 và script huấn luyện có giám sát. |
| **L6** | **TabularQAgent / DQNAgent** | Học tăng cường thông qua cơ chế tự chơi (Self-Play): Q-Learning dạng bảng cho $3 \times 3, K=3$; Deep Q-Network (DQN) với Replay Buffer, Target Network và $\epsilon$-greedy schedule cho $4 \times 4$. | [`src/ttt_ai/agents/rl_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/rl_agent.py) | **Khung đã thiết kế (Sprint 3)**<br>Sẵn sàng quy trình train đa seed (3 seeds) với opponent pool hỗn hợp. |
| **L7** | **HybridMCTSAgent** | Mô hình kết hợp lấy cảm hứng từ kiến trúc AlphaGo: MCTS sử dụng biến thể **PUCT**, ưu tiên duyệt nhánh được định hướng bởi `StrongPolicyNet`, kết hợp giá trị lá từ `ValueNet` và Rollouts. | [`src/ttt_ai/agents/hybrid_mcts_agent.py`](Advanced_AI_Tic-Tac-Toe_Project/src/ttt_ai/agents/hybrid_mcts_agent.py) | **Khung đã thiết kế (Sprint 3)**<br>Đặc tả đầy đủ tại `ARCHITECTURE.md` (giải quyết câu hỏi nghiên cứu trọng tâm **RQ3**). |

---

## ⚙️ Cài đặt & Môi trường (Prerequisites & Installation)

### Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+, hoặc macOS.
- **Python**: Khuyến nghị phiên bản **Python 3.11** hoặc **Python 3.12** (Đã kiểm thử tương thích tốt với Python 3.14).
- **Phần cứng**: Tối thiểu 4GB RAM. Khuyến nghị có GPU NVIDIA hỗ trợ **CUDA** để huấn luyện mạng sâu nhanh hơn (tùy chọn, CPU vẫn chạy mượt mà do kích thước không gian cờ nhỏ gọn).

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

#### Bước 3: Cài đặt thư viện thực hành Lab
Cài đặt các gói phụ thuộc dùng cho các bài tập thực hành Jupyter Notebook (Lab 03 & Lab 04):
```bash
pip install numpy scipy matplotlib pandas numba tabulate jupyter
```

#### Bước 4: Cài đặt package Đồ án Game AI (`ttt-ai`)
Cài đặt package đồ án ở chế độ phát triển (editable mode) kèm các gói công cụ dev (`pytest`, `ruff`):
```bash
cd Advanced_AI_Tic-Tac-Toe_Project
pip install -e .[dev]
cd ..
```

---

## 🚀 Hướng dẫn chạy (Usage & Execution)

### 1. Chạy các bài thực hành Lab (Jupyter Notebooks)

Di chuyển vào thư mục bài Lab tương ứng và khởi động Jupyter Lab / Notebook.  
> **Lưu ý quan trọng**: Phải chạy Jupyter từ chính thư mục của từng bài Lab để đường dẫn đọc file ma trận (`L_maze.txt`) và module hỗ trợ (`maze_helper.py`) được xác thực chính xác.

```bash
# Khởi chạy Lab 03 (Local Search)
cd Lab/lab03_localsearch_search_TruongChiLong_3123410198
jupyter notebook

# Hoặc khởi chạy Lab 04 (CSP)
cd ../lab04_csp_TruongChiLong_3123410198
jupyter notebook
```

### 2. Chạy kiểm thử tự động (Unit Tests)

Dự án xây dựng bộ kiểm thử tự động toàn diện với **216 test cases** bao phủ toàn bộ động cơ trò chơi và các tác tử AI:

```bash
# Chạy toàn bộ 216 unit tests trong repository
pytest Advanced_AI_Tic-Tac-Toe_Project -v

# Chạy riêng bộ kiểm thử Động cơ Game Engine & Luật kiểm tra thắng O(K)
pytest Advanced_AI_Tic-Tac-Toe_Project/tests/test_engine.py Advanced_AI_Tic-Tac-Toe_Project/tests/test_rules.py -v

# Chạy kiểm thử các tác tử tìm kiếm Minimax, Alpha-Beta và MCTS
pytest Advanced_AI_Tic-Tac-Toe_Project/tests/test_search_agents.py -v

# Chạy kiểm thử tuân thủ hợp đồng giao tiếp tác tử (Agent Contract)
pytest Advanced_AI_Tic-Tac-Toe_Project/tests/test_agents_contract.py -v
```

### 3. Chạy linter kiểm tra chuẩn mã nguồn

Dự án áp dụng quy chuẩn PEP 8 và linter **Ruff** nghiêm ngặt:
```bash
ruff check Advanced_AI_Tic-Tac-Toe_Project/src Advanced_AI_Tic-Tac-Toe_Project/tests
```

### 4. Khởi chạy Game AI & Scripts Đồ án

Khi di chuyển vào thư mục `Advanced_AI_Tic-Tac-Toe_Project/`, các kịch bản thực nghiệm và giao diện CLI được vận hành như sau:

```bash
cd Advanced_AI_Tic-Tac-Toe_Project

# 1. Trực tiếp chơi đối kháng với AI trên Terminal giao diện Rich CLI (Human vs AI hoặc AI vs AI)
python scripts/play.py --config configs/game_4x4_k4.yaml --agent alphabeta --depth 4

# 2. Khởi chạy giải đấu Paired Round-Robin giữa các cấp độ tác tử AI
python scripts/run_tournament.py --config configs/tournament.yaml

# 3. Chạy 4 kịch bản phân tích thực nghiệm cô lập (Ablation Studies RQ3, RQ4, RQ5)
python scripts/run_ablations.py --output reports/

# 4. Xuất 5 bộ biểu đồ báo cáo và thống kê tự động
python scripts/plot_results.py --input logs/ --output reports/
```

---

## 📊 Kết quả & Đánh giá (Experiments & Evaluation)

### 1. Thực nghiệm Tìm kiếm cục bộ (Lab 03)
- **Mê cung chữ L (Local Optima)**: Thuật toán *Simple Hill Climbing* xuất phát từ vị trí $S(9, 5)$ với khoảng cách Manhattan $h=14$. Do tường chữ L cản trở, thuật toán luôn rơi vào bẫy cực trị cục bộ tại tọa độ $(5, 9)$ có $h=6$ và không thể tìm thấy đích đến $G(2, 12)$, bất kể sử dụng thứ tự hướng di chuyển nào (`NESW`, `ESWN`,...).
- **Đánh giá các biến thể Hill Climbing trên $n$-Queens ($n=8$)**:
  - Các phương pháp leo đồi đơn lẻ (*Steepest-Ascent*, *Stochastic*) chỉ đạt tỷ lệ thành công khiêm tốn: **12.0% – 14.0%**. Tuy nhiên, *Efficient Stochastic HC (First-choice)* giảm hơn **55% thời gian thực thi** so với *Steepest-Ascent*.
  - *Random Restart Hill Climbing* đạt tỷ lệ thành công tuyệt đối **100%** với số lần khởi động lại trung bình là **6.8 lần**.
  - *Simulated Annealing* với lịch Exponential ($T_0=4.0, \alpha=0.995$) đạt tỷ lệ thành công ấn tượng **98.0%**.
  - Trong 4 toán tử đề xuất, `move_adaptive` cho tốc độ hội tụ nhanh và ổn định nhất.
- **Tăng tốc Numba JIT**: Khảo sát bài toán $n$-Queens ở quy mô lớn ($n=450$) cho thấy phiên bản biên dịch Numba JIT `@njit` tính toán xung đột nhanh hơn phiên bản tuyến tính Python thuần $O(n)$ gấp **90.3 lần**, và vượt trội hơn thuật toán Naive $O(n^2)$ hàng nghìn lần.

### 2. Thực nghiệm Bài toán thỏa mãn ràng buộc (Lab 04)
- **Tô màu đồ thị phẳng (Graph Coloring)**:
  - Với không gian 4 màu ($k=4, n=10$ đỉnh): Cả Backtracking cơ bản, Backtracking + MRV/LCV, Forward Checking và Min-Conflicts đều tìm thấy lời giải hợp lệ trong 100% số lần chạy (trung bình chỉ cần 8 đến 11 bước mở rộng).
  - Với không gian 3 màu ($k=3$): Thuật toán Min-Conflicts đạt tỷ lệ thành công 0% do vướng vào cực trị địa phương khi sắc số đồ thị $\chi(G) > 3$.
- **Khảo sát độ mở rộng (Scalability Analysis)**: Phân tích Log-Log cho thấy độ dốc biểu diễn chi phí thời gian của *Simulated Annealing* ($1.06$) phẳng hơn đáng kể so với *Efficient Stochastic HC* ($1.92$), khẳng định khả năng giải quyết bài toán quy mô lớn vượt trội của kỹ thuật làm lạnh mô phỏng.

### 3. Kiểm thử & Sanity Benchmark Đồ án Game AI (Sprint 1 & Sprint 2)

Dự án đã hoàn thành vượt mức tiến độ Sprint 1 và Sprint 2, đạt chứng chỉ kiểm thử và dữ liệu benchmark thực tế:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.x, pytest-9.2.0, pluggy-1.5.0
rootdir: C:\Users\Ron\Documents\GitHub\Advanced-AI-03\Advanced_AI_Tic-Tac-Toe_Project
configfile: pyproject.toml
testpaths: tests
collected 216 items

tests/test_config.py ........................                            [ 11%]
tests/test_rules.py ...........................                          [ 23%]
tests/test_engine.py .............................                       [ 37%]
tests/test_base_agent.py ...........................                     [ 49%]
tests/test_level1_level2_agents.py ................                      [ 56%]
tests/test_minimax_agent.py ..............                               [ 63%]
tests/test_mcts_agent.py ..................................              [ 79%]
tests/test_agents_contract.py ......................                     [ 89%]
tests/test_search_agents.py .......................                      [100%]

============================= 216 passed in 21.01s =============================
```

#### Dữ liệu Benchmark Thực tế giữa các Cặp tác tử (Sanity Runs):
1. **Hiệu năng Cắt tỉa Alpha-Beta + Move Ordering (`4×4, K=4`, Bàn cờ rỗng, Độ sâu $d=4$)**:
   - *Minimax thuần (Duyệt cạn)*: Duyệt **47,297** nút (Thời gian: $523.16$ ms).
   - *Alpha-Beta (Không sắp xếp nước đi)*: Duyệt **3,169** nút (Thời gian: $89.17$ ms).
   - *Alpha-Beta + Move Ordering*: Duyệt duy nhất **1,941** nút (Thời gian: $50.67$ ms).
   - 👉 **Kết luận**: Alpha-Beta kết hợp heuristic Move Ordering **giảm tới 95.9% số nút cần duyệt** mà vẫn bảo toàn giá trị lượng giá và nước đi tối ưu tuyệt đối (`action=0`, `val=-0.008000`).
2. **Giải đấu Vòng tròn Sanity Round-Robin 4 Cấp độ (`4×4, K=4`, 120 ván đối kháng)**:
   - **Cấp 1 (`RandomAgent`)**: Thắng 0, Hòa 9, Thua 51 (Score: `0.075`, $0.028$ ms/nước).
   - **Cấp 2 (`RuleBasedAgent`)**: Thắng 18, Hòa 42, Thua 0 (Score: `0.650`, $0.065$ ms/nước).
   - **Cấp 3 (`AlphaBetaAgent d=4`)**: Thắng 15, Hòa 45, Thua 0 (Score: `0.625`, $10.71$ ms/nước).
   - **Cấp 4 (`MCTSAgent N=200`)**: Thắng 18, Hòa 42, Thua 0 (Score: `0.650`, $15.86$ ms/nước).
   - 👉 **Kết luận**: Cả ba tác tử Cấp 2, Cấp 3 và Cấp 4 đều giữ vững thành tích **bất bại (0 trận thua)** trước RandomAgent và hòa nhau cân bằng tuyệt đối (`0-20-0`, Score `0.500`) trong các cặp đấu đối đầu trực tiếp.

---

## 📚 Nguồn Tham khảo & Kế thừa (References & Inheritance)

### 1. Tài liệu môn học & Đề tài (GVHD: Đỗ Như Tài)
- [`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf) — Đề cương chi tiết và yêu cầu kỹ thuật đồ án cuối kỳ môn Trí Tuệ Nhân Tạo Nâng Cao.
- [`Tic tac toe Game.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/Tic%20tac%20toe%20Game.pdf) — Slide hướng dẫn nguyên lý trò chơi Tic-Tac-Toe mở rộng và phương pháp tìm kiếm trong không gian trạng thái.

### 2. Giáo trình & Tài liệu kinh điển
- **Stuart Russell & Peter Norvig**, *Artificial Intelligence: A Modern Approach (4th Edition)*, Pearson, 2020. (Nền tảng về Không gian trạng thái, Heuristic Search, CSP, Minimax và Games).
- **Richard S. Sutton & Andrew G. Barto**, *Reinforcement Learning: An Introduction (2nd Edition)*, The MIT Press, 2018. (Nền tảng về MDP, Bellman Equations, Q-Learning và Deep RL).
- **Mark Liu**, *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*, 2024. Tài liệu và mã nguồn đối chiếu tại [`AlphaGoSimplified-main/`](Advanced_AI_Tic-Tac-Toe_Project/References/AlphaGoSimplified-main/).

### 3. Mã nguồn tham khảo & Khung thực hành
- [`maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) và khung bài tập thực hành Lab 03 / Lab 04 của **Michael Hahsler** (Khoa Khoa học Máy tính, SMU). Toàn bộ phần thuật toán tìm kiếm, tối ưu hóa Numba, đối sánh thực nghiệm và báo cáo chuyên sâu do sinh viên thực hiện độc lập.
- [`Reversi-mcts-master/`](Advanced_AI_Tic-Tac-Toe_Project/References/Reversi-mcts-master/) — Dự án mã nguồn C#/.NET tham khảo phương pháp áp dụng MCTS và Backtracking Minimax trong trò chơi đối kháng bàn cờ.

---

## 👤 Tác giả & Giấy phép (Author & License)

- **Tác giả:** Trương Chí Long
- **Mã số sinh viên:** `3123410198`
- **Lớp / Nhóm:** Nhóm 03 — Môn học: Trí Tuệ Nhân Tạo Nâng Cao (`841458`)
- **Khoa / Trường:** Khoa Công nghệ Thông tin — Trường Đại học Sài Gòn (SGU)
- **Giảng viên hướng dẫn:** ThS. Đỗ Như Tài
- **Bản quyền & Giấy phép:** Dự án được phát hành theo giấy phép [MIT License](https://opensource.org/licenses/MIT). Toàn bộ nội dung học thuật, mã nguồn và tài liệu tuân thủ quy chuẩn trung thực học thuật (*Zero Fabrication & Reproducibility*).
