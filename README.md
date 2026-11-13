# Advanced-AI-03 — Trí Tuệ Nhân Tạo Nâng Cao (841458)

Kho lưu trữ bài tập lý thuyết, bài thực hành và đồ án cuối kỳ môn **Trí Tuệ Nhân Tạo Nâng Cao**.

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

1. [Tổng quan Cấu trúc Repository](#1-tổng-quan-cấu-trúc-repository)
2. [Bài tập Lý thuyết (`Ex/`)](#2-bài-tập-lý-thuyết-ex)
3. [Bài tập Thực hành (`Lab/`)](#3-bài-tập-thực-hành-lab)
4. [Đồ án Cuối kỳ: Game AI Đa Cấp Độ (`Advanced_AI_Tic-Tac-Toe_Project/`)](#4-đồ-án-cuối-kỳ-game-ai-đa-cấp-độ-advanced_ai_tic-tac-toe_project)
5. [Hướng dẫn Cài đặt & Chạy thử (Hiện trạng Hiện tại)](#5-hướng-dẫn-cài-đặt--chạy-thử-hiện-trạng-hiện-tại)
6. [Nguồn Tham khảo & Kế thừa](#6-nguồn-tham-khảo--kế-thừa)

---

## 1. Tổng quan Cấu trúc Repository

```text
Advanced-AI-03/
├── README.md                                        # Tài liệu tổng quan toàn repository
├── Ex/                                              # Bài tập lý thuyết (PDF)
│   ├── ex01_AISearch_TruongChiLong_3123410198.pdf
│   ├── ex02_CSPGame_TruongChiLong_3123410198.pdf
│   └── ex03_AIGame_TruongChiLong_3123410198.pdf
├── Lab/                                             # Bài thực hành lập trình & báo cáo
│   ├── lab03_localsearch_search_TruongChiLong_3123410198/
│   │   ├── README.md
│   │   ├── BAOCAO_lab03_localsearch_search.pdf
│   │   ├── L_maze.txt
│   │   ├── maze_helper.py
│   │   ├── Maze_Example_LocalSearch.ipynb
│   │   ├── n_queens.ipynb
│   │   ├── n_queens_fast_conflict_calculation_with_numba.ipynb
│   │   ├── Simulated_Annealing_Schedules.ipynb
│   │   └── traveling_salesman_problem.ipynb
│   └── lab04_csp_TruongChiLong_3123410198/
│       ├── README.md
│       ├── lab04_csp_TruongChiLong_3123410198.pdf
│       ├── CSP_graph_coloring_example.ipynb
│       ├── n_queens.ipynb
│       ├── Graph Coloring.png
│       ├── Queen.png
│       ├── Algorithm Convergence.png
│       └── Runtime Scalability.png
└── Advanced_AI_Tic-Tac-Toe_Project/                 # Đồ án cuối kỳ: Extended Tic-Tac-Toe (M×N, K)
    ├── Planning/                                    # 6 tài liệu thiết kế & theo dõi (Đã hoàn tất)
    │   ├── PRD.md
    │   ├── RULES.md
    │   ├── ARCHITECTURE.md
    │   ├── DESIGN.md
    │   ├── TASKS.md
    │   └── PROGRESS.md
    ├── References/                                  # Đề cương, slide, sách & repo tham khảo
    ├── configs/                                     # Khung tệp YAML (chưa điền nội dung)
    ├── src/ttt_ai/                                  # Khung mã nguồn package (chưa điền nội dung)
    ├── scripts/                                     # Khung CLI scripts (chưa điền nội dung)
    ├── tests/                                       # Khung unit tests (chưa điền nội dung)
    ├── checkpoints/                                 # Thư mục lưu trọng số (trống)
    ├── logs/                                        # Thư mục lưu log thực nghiệm (trống)
    ├── reports/                                     # Thư mục xuất biểu đồ/bảng (trống)
    ├── pyproject.toml                               # Khung cấu hình package (trống)
    └── README.md                                    # Khung README đồ án (trống)
```

### Tóm tắt Trạng thái Hoàn thành

| Thành phần | Thư mục | Trạng thái |
|---|---|---|
| **Bài tập lý thuyết (`ex01` – `ex03`)** | [`Ex/`](Ex/) | Đã hoàn thành (`3/3` PDF) |
| **Thực hành Lab 03 (`Local Search`)** | [`Lab/lab03_localsearch_search_TruongChiLong_3123410198/`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/) | Đã hoàn thành (`5` notebooks + báo cáo PDF) |
| **Thực hành Lab 04 (`CSP`)** | [`Lab/lab04_csp_TruongChiLong_3123410198/`](Lab/lab04_csp_TruongChiLong_3123410198/) | Đã hoàn thành (`2` notebooks + 4 hình + báo cáo PDF) |
| **Đồ án cuối kỳ (`Extended Tic-Tac-Toe`)** | [`Advanced_AI_Tic-Tac-Toe_Project/`](Advanced_AI_Tic-Tac-Toe_Project/) | Hoàn tất **Phase 0 (`Planning/` 6/6)**; **Sprint 1–5** (`configs/`, `src/`, `scripts/`, `tests/`) đang ở trạng thái khởi tạo khung rỗng |

---

## 2. Bài tập Lý thuyết (`Ex/`)

Thư mục [`Ex/`](Ex/) lưu trữ các bài giải lý thuyết dạng PDF về tìm kiếm, bài toán thỏa mãn ràng buộc (CSP) và lý thuyết trò chơi:

| Tệp | Chủ đề chính | Nội dung đã giải quyết |
|---|---|---|
| [`ex01_AISearch_TruongChiLong_3123410198.pdf`](Ex/ex01_AISearch_TruongChiLong_3123410198.pdf) | Tìm kiếm & Heuristic trong AI | Mô phỏng **BFS, DFS, IDS, UCS, Greedy Search, A\***; mô hình hóa không gian trạng thái bài toán Tháp Hà Nội ($\|S\| = 3^n$); phân tích tính *admissible* và *consistent* của hàm heuristic (bao gồm bài toán điều hướng có vận tốc $v$ và khoảng cách Manhattan); khảo sát tập nút được mở rộng và các nhận định đúng/sai về Tree/Graph Search. |
| [`ex02_CSPGame_TruongChiLong_3123410198.pdf`](Ex/ex02_CSPGame_TruongChiLong_3123410198.pdf) | CSP & Game trong AI | Nhận diện cực trị cục bộ (*local optimum*) và vai bằng (*shoulder*) trong Local Search; **Minimax** & **Alpha-Beta Pruning** (ảnh hưởng thứ tự duyệt và iterative deepening); **Expectiminimax** và quy tắc cắt tỉa khi biết cận trên/dưới của lá; so sánh **Forward Checking** và **AC-3**, heuristic **MRV** & **LCV** trên các bài toán lập lịch đường băng, ma phương 3×3 và tô màu đồ thị $K_3$. |
| [`ex03_AIGame_TruongChiLong_3123410198.pdf`](Ex/ex03_AIGame_TruongChiLong_3123410198.pdf) | Tìm kiếm, CSP & Game nâng cao | Tính giá trị cây **Minimax** và tỉa nhánh **Alpha-Beta**; cắt tỉa trong cây **Expectiminimax**; phân tích chính sách Minimax vs Expectimax vs **Value Iteration (MDP)** trong môi trường ngẫu nhiên; lan truyền **AC-3** trên đồ thị và thứ tự gán biến từ gốc tới lá so với từ lá lên gốc; trò chơi hai hàm mục tiêu và hợp tác hoàn toàn (Pacman–Casper); điều kiện đầy đủ/tối ưu của BFS/DFS/IDS/A\*; cập nhật **A\*** khi chi phí cạnh thay đổi; và phân loại lớp tác tử hợp lý. |

---

## 3. Bài tập Thực hành (`Lab/`)

### 3.1. Lab 03 — Tìm kiếm cục bộ (Local Search)

Thư mục: [`Lab/lab03_localsearch_search_TruongChiLong_3123410198/`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/)  
Tài liệu kèm theo: [README Lab 03](Lab/lab03_localsearch_search_TruongChiLong_3123410198/README.md) | [Báo cáo PDF (`BAOCAO_lab03_localsearch_search.pdf`)](Lab/lab03_localsearch_search_TruongChiLong_3123410198/BAOCAO_lab03_localsearch_search.pdf)

| Tệp thực hành | Bài toán & Kỹ thuật triển khai | Kết quả / Ghi nhận nổi bật từ mã nguồn & báo cáo |
|---|---|---|
| [`Maze_Example_LocalSearch.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Maze_Example_LocalSearch.ipynb) + [`maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) + [`L_maze.txt`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/L_maze.txt) | **Simple Hill Climbing trên Mê cung chữ L** với hàm heuristic khoảng cách Manhattan $h(n) = \|r_n - r_G\| + \|c_n - c_G\|$; thử nghiệm nhiều thứ tự hướng (`NESW`, `ESWN`, `SWNE`, `WNES`) và chế độ ngẫu nhiên. | Xuất phát từ `S(9, 5)` ($h=14$), thuật toán luôn dừng tại cực trị cục bộ `[5, 9]` ($h=6$) phía sau bức tường chữ L mà không tới được đích `G(2, 12)`. |
| [`n_queens.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens.ipynb) | **Bài toán n-Queens**: cài đặt **Steepest-Ascent HC**, **Inefficient Stochastic HC**, **Efficient Stochastic (First-choice) HC**, **Random Restart HC** và **Simulated Annealing**; khảo sát hội tụ & khả năng mở rộng ($n \in \{4, 8, 12, 16, 20\}$); **Advanced Task** với 4 toán tử (`move_single_step`, `move_column_swap`, `move_dual_queen`, `move_adaptive`) và phần mở rộng **Genetic Algorithm** (`genetic_algorithm_nqueens`). | Trên $n=8$: các biến thể HC đơn lẻ đạt tỷ lệ thành công ~12%–14% (Efficient Stochastic HC nhanh hơn hơn 55% thời gian chạy so với Steepest-Ascent); Random Restart đạt 100% thành công (TB ~6.8 lần restart); Simulated Annealing (`T0=4.0, alpha=0.995`) đạt 98.0%. Trong 4 toán tử mới, `move_adaptive` hội tụ nhanh và đạt tỷ lệ thành công cao nhất. |
| [`Simulated_Annealing_Schedules.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/Simulated_Annealing_Schedules.ipynb) | **Khảo sát 4 lịch giảm nhiệt độ** cho Simulated Annealing: *Exponential*, *Linear*, *Fast* và *Logarithmic*. | Đánh giá đường cong giảm nhiệt $T(t)$ và giải thích đánh đổi giữa *Exploration* và *Exploitation* (lịch *Exponential* $\alpha \in [0.99, 0.995]$ cân bằng tốt nhất trong số vòng lặp hữu hạn). |
| [`n_queens_fast_conflict_calculation_with_numba.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/n_queens_fast_conflict_calculation_with_numba.ipynb) | **Tối ưu hàm đếm xung đột n-Queens**: so sánh `conflicts_naive` $O(n^2)$, `conflicts` $O(n)$ theo tổ hợp $\binom{k}{2}$ trên hàng/chéo, và `conflicts_jit` biên dịch bằng **Numba (`@njit`)**. | Khử thời gian warm-up biên dịch JIT lần đầu; trên notebook đo ở $n=450$, `conflicts_jit` nhanh hơn bản Linear Python $O(n)$ khoảng `90.3x` và nhanh hơn bản Naive $O(n^2)$ tới hàng nghìn lần (tùy cấu hình phần cứng chạy trong notebook/báo cáo). |
| [`traveling_salesman_problem.ipynb`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/traveling_salesman_problem.ipynb) | **Traveling Salesman Problem (TSP)** với 5 thuật toán Local Search + **Bonus Genetic Algorithm** (`order_crossover`, `tsp_genetic_algorithm`); so sánh toán tử lân cận `2-swap` và đảo đoạn `2-opt`. | Đánh giá trên các kích thước $n \in \{10, 20, 30, 50\}$; toán tử `2-opt` tháo gỡ trực tiếp các cạnh giao cắt trên mặt phẳng Euclid nên cho chu trình ngắn hơn đáng kể so với `2-swap`. |

### 3.2. Lab 04 — Bài toán thỏa mãn ràng buộc (CSP)

Thư mục: [`Lab/lab04_csp_TruongChiLong_3123410198/`](Lab/lab04_csp_TruongChiLong_3123410198/)  
Tài liệu kèm theo: [README Lab 04](Lab/lab04_csp_TruongChiLong_3123410198/README.md) | [Báo cáo PDF (`lab04_csp_TruongChiLong_3123410198.pdf`)](Lab/lab04_csp_TruongChiLong_3123410198/lab04_csp_TruongChiLong_3123410198.pdf)

| Tệp thực hành | Nội dung & Thuật toán đã cài đặt | Kết quả & Trực quan hóa |
|---|---|---|
| [`CSP_graph_coloring_example.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/CSP_graph_coloring_example.ipynb) | **Tô màu đồ thị (Graph Coloring)** sinh bằng Delaunay triangulation (`scipy.spatial.Delaunay`); kiểm tra `complete()` / `consistent()`; **Backtracking Search** cơ bản và phần Assignment: **MRV** (`select_unassigned_var_mrv`), **LCV** (`order_domain_values_lcv`), **Forward Checking** (`backtrack_search_fc`), và **Min-Conflicts** (`min_conflicts`). | Với $n=10$ và 4 màu: Backtracking, MRV+Backtracking, Forward Checking và Min-Conflicts đều đạt 100% nghiệm hợp lệ (10–11 nút / TB 8.0 bước). Với 3 màu: Min-Conflicts đạt 0% (kẹt do sắc số đồ thị $> 3$). Hình kết quả: [`Graph Coloring.png`](Lab/lab04_csp_TruongChiLong_3123410198/Graph%20Coloring.png). |
| [`n_queens.ipynb`](Lab/lab04_csp_TruongChiLong_3123410198/n_queens.ipynb) | **n-Queens (CSP / Local Search)**: các biến thể Hill Climbing, Random Restart, Simulated Annealing, phân tích hội tụ & hệ số góc Log-Log (`Slope` của Efficient Stochastic HC `1.92` vs Simulated Annealing `1.06`), 4 toán tử (`move_single_step`, `move_col_swap`, `move_dual_queen`, `move_adaptive`) và **Genetic Algorithm** (`genetic_algorithm`). | Kèm các hình xuất trực tiếp: [`Queen.png`](Lab/lab04_csp_TruongChiLong_3123410198/Queen.png), [`Algorithm Convergence.png`](Lab/lab04_csp_TruongChiLong_3123410198/Algorithm%20Convergence.png), [`Runtime Scalability.png`](Lab/lab04_csp_TruongChiLong_3123410198/Runtime%20Scalability.png). |

> **Lưu ý môi trường:** Thư mục ảo `Lab/lab04_csp_TruongChiLong_3123410198/.venv/` nằm trong `.gitignore` (chỉ tồn tại cục bộ, không được theo dõi trên Git).

---

## 4. Đồ án Cuối kỳ: Game AI Đa Cấp Độ (`Advanced_AI_Tic-Tac-Toe_Project/`)

Dự án cuối kỳ xây dựng và đánh giá hệ thống Game AI đa cấp độ cho trò chơi **Extended Tic-Tac-Toe (`M×N, K`)**, tuân theo đề cương [`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf) và định hướng từ [`Tic tac toe Game.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/Tic%20tac%20toe%20Game.pdf).

### Trạng thái triển khai thực tế
- **Phase 0 (`Planning/`): ĐÃ HOÀN TẤT (`6/6` tài liệu)** — Đã đặc tả đầy đủ yêu cầu, quy chuẩn kỹ thuật/trung thực học thuật, kiến trúc phân lớp, giao diện Rich CLI/biểu đồ và kế hoạch 5 Sprint.
- **Sprint 1 – Sprint 5 (`configs/`, `src/ttt_ai/`, `scripts/`, `tests/`, `pyproject.toml`, `README.md`): ĐANG Ở TRẠNG THÁI `TODO`** — Các tệp cấu hình, mã nguồn và unit tests hiện đã được tạo cấu trúc cây thư mục nhưng đang có dung lượng `0 bytes`, chưa cài đặt logic và chưa có dữ liệu checkpoint/log/report thực nghiệm.

### Tài liệu Thiết kế (`Planning/`)

| Tài liệu | Vai trò |
|---|---|
| [`PRD.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/PRD.md) | Đặc tả bài toán `M×N, K` (chuẩn chính `4×4, K=4`, biến thể `4×4, K=3`, đối chứng `3×3, K=3`, mở rộng `5×5, K=4`), mô hình MDP, 7 cấp tác tử AI, thiết kế giải đấu và **3 câu hỏi nghiên cứu (`RQ3`, `RQ4`, `RQ5`)**, ánh xạ Rubric 100 điểm (+10 điểm khuyến khích). |
| [`RULES.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/RULES.md) | Quy định trung thực học thuật (*Zero Fabrication*, *No Cherry-Picking*, cấm dùng trọng số chưa train, chia dữ liệu theo `game_id`), chuẩn Game API, đo ngân sách tính toán công bằng (`time.perf_counter()`, đa seed, Wilson/Bootstrap 95% CI) và chuẩn PEP 8 / Type Hints. |
| [`ARCHITECTURE.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/ARCHITECTURE.md) | Kiến trúc tách lớp `engine` (thuần Python + NumPy) – `agents` – `training` – `benchmark` – `ui`; API `ExtendedTicTacToe` (`reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`/`undo`); chi tiết 7 cấp AI (`Random`, `RuleBased` 6 tầng luật, `Minimax/Alpha-Beta` + Iterative Deepening, `MCTS UCT`, `FastPolicyNet`/`StrongPolicyNet`/`ValueNet`, `TabularQ`/`DQN Self-Play`, `HybridMCTSAgent` PUCT); schema `meta.json` và JSONL logs. |
| [`DESIGN.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/DESIGN.md) | Thiết kế giao diện terminal **Rich CLI** (`Setup Wizard`, `Live Match Dashboard` với Board, AI Telemetry, Action Heatmap & Top-5 Moves, hỗ trợ `Human vs AI` và `AI vs AI`) và hệ thống 5 biểu đồ báo cáo tự động (`Fig 1`–`Fig 5`). |
| [`TASKS.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/TASKS.md) | Phân rã chi tiết 5 Sprint (`T1.1`–`T5.3`) kèm tiêu chí hoàn thành (DoD) và cổng nghiệm thu cho từng Sprint. |
| [`PROGRESS.md`](Advanced_AI_Tic-Tac-Toe_Project/Planning/PROGRESS.md) | Bảng theo dõi tiến độ thực tế, `Experiment & Checkpoint Registry` và `Bug & Issue Log`. |

### Các Trọng tâm Thiết kế của Đồ án (theo `Planning/`)
- **7 cấp độ tác tử AI dự kiến:**
  1. `RandomAgent` (baseline ngẫu nhiên)
  2. `RuleBasedAgent` (tổng quát hóa luật Newell & Simon theo 6 mức ưu tiên)
  3. `MinimaxAgent` / `AlphaBetaAgent` (Iterative Deepening theo `time_budget_ms`, `precompute_win_windows`)
  4. `MCTSAgent` (UCT 4 pha: Selection, Expansion, Simulation, Backpropagation)
  5. `PolicyNetworkAgent` / `ValueGreedyAgent` (`FastPolicyNet`, `StrongPolicyNet`, `ValueNet` có action masking)
  6. `RLAgent` (`TabularQAgent` cho `3×3, K=3`; `DQNAgent` Self-Play + Mixed Opponent Pool cho `4×4`)
  7. `HybridMCTSAgent` (PUCT dẫn hướng bởi `StrongPolicyNet`, LRU cache theo canonical board, tùy chọn `rollout` / `mixed` với `ValueNet` — mô phỏng cảm hứng từ AlphaGo)
- **3 Câu hỏi nghiên cứu trọng tâm:**
  - **RQ3:** Hiệu quả của `Hybrid MCTS + Policy/Value` so với `Pure MCTS` và `Policy-only` trên `4×4, K=4` theo ngân sách mô phỏng và độ trễ.
  - **RQ4:** Độ nhạy lợi thế người đi trước ($\Delta\text{Score} = \text{Score}_X - \text{Score}_O$) theo cấp độ AI và điều kiện thắng `K` (`3×3 K=3`, `4×4 K=3`, `4×4 K=4`).
  - **RQ5:** Độ ổn định qua nhiều seed và khả năng tổng quát hóa trước đối thủ *unseen* giữa `RL Self-play` và `Supervised Learning`.

---

## 5. Hướng dẫn Cài đặt & Chạy thử (Hiện trạng Hiện tại)

### 5.1. Clone repository
```bash
git clone https://github.com/ronstudy/Advanced-AI-03.git
cd Advanced-AI-03
```

### 5.2. Chạy các Notebook Thực hành (`Lab/lab03` & `Lab/lab04`)
Tạo môi trường ảo Python và cài đặt các thư viện được sử dụng trong hai bài Lab (`numpy`, `matplotlib`, `scipy`, `pandas`, `numba`, `jupyter`, `tabulate`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib scipy pandas numba tabulate jupyter
```

Sau đó mở Jupyter Notebook tại thư mục bài Lab tương ứng (chú ý giữ đúng thư mục làm việc để `Maze_Example_LocalSearch.ipynb` nạp được [`maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) và [`L_maze.txt`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/L_maze.txt)):

```powershell
cd Lab\lab03_localsearch_search_TruongChiLong_3123410198
jupyter notebook
```

### 5.3. Đối với `Advanced_AI_Tic-Tac-Toe_Project/`
Dự án hiện mới hoàn tất bộ tài liệu thiết kế trong [`Planning/`](Advanced_AI_Tic-Tac-Toe_Project/Planning/) (các tệp `pyproject.toml`, `configs/*.yaml`, `src/ttt_ai/`, `scripts/`, `tests/` hiện đang trống chờ triển khai từ Sprint 1). Sau khi hoàn thiện Sprint 1–5, hướng dẫn cài đặt `pip install -e .`, chạy `pytest`, huấn luyện và khởi chạy `scripts/play.py` sẽ được cập nhật chi tiết tại [`Advanced_AI_Tic-Tac-Toe_Project/README.md`](Advanced_AI_Tic-Tac-Toe_Project/README.md) và đồng bộ về đây.

---

## 6. Nguồn Tham khảo & Kế thừa

- **Tài liệu môn học & bài giảng (GVHD: Đỗ Như Tài):**
  - [`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/TTNTNC_De_tai_cuoi_ky_Game_AI.pdf)
  - [`Tic tac toe Game.pdf`](Advanced_AI_Tic-Tac-Toe_Project/References/Tic%20tac%20toe%20Game.pdf)
- **Giáo trình tham khảo trong [`References/`](Advanced_AI_Tic-Tac-Toe_Project/References/):**
  - Mark Liu (2024), *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games* (PDF & mã nguồn đối chiếu [`AlphaGoSimplified-main/`](Advanced_AI_Tic-Tac-Toe_Project/References/AlphaGoSimplified-main/))
  - Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach* (4th Edition)
  - Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd Edition)
- **Mã nguồn tham khảo & khung thực hành:**
  - [`maze_helper.py`](Lab/lab03_localsearch_search_TruongChiLong_3123410198/maze_helper.py) và khung bài tập Lab 03 / Lab 04 (Michael Hahsler — khung bài tập *Local Search* & *CSP*, phần thuật toán giải, tối ưu Numba, khảo sát thực nghiệm và báo cáo do sinh viên thực hiện).
  - [`Reversi-mcts-master/`](Advanced_AI_Tic-Tac-Toe_Project/References/Reversi-mcts-master/) — kho mã nguồn C#/.NET tham khảo về MCTS & BTMM.
