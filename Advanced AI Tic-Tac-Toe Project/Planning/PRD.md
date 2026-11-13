# PRD — Xây dựng và Đánh giá Game AI Đa Cấp Độ (Extended Tic-Tac-Toe)

- **Phiên bản:** `1.0`
- **Môn học:** Trí tuệ nhân tạo nâng cao (Advanced AI)
- **Tài liệu tham chiếu nền tảng:**
  - Đề cương môn học: `TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`
  - Slide thiết kế thuật toán: `Tic tac toe Game.pdf`
  - Sách tham khảo chính: Mark Liu (2024), *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*

---

## 1. Tổng quan & Mục tiêu dự án

### 1.1. Bối cảnh & Định vị bài toán
Dự án xây dựng một hệ thống trò chơi **Tic-Tac-Toe mở rộng (`M×N, K`)** hoàn chỉnh theo mô hình môi trường ra quyết định đối kháng, đi kèm **7 cấp độ tác tử AI** từ cơ bản đến học sâu/tăng cường và mô hình lai (Hybrid), một hệ thống **giải đấu đối kháng công bằng**, và **giao diện Rich CLI** trực quan cho người dùng lẫn trình diễn học thuật.

Trên bàn `3×3` truyền thống, không gian trạng thái nhỏ khiến các tác tử mạnh (Rule-based chuẩn, Alpha-Beta toàn cây) dễ kéo nhau về thế hòa và làm mờ khác biệt giữa các phương pháp học máy. Do đó, dự án thiết kế Engine tổng quát `M×N, K` và chọn **`4×4, K=4`** làm chuẩn thực nghiệm chính để làm bộc lộ rõ bài toán nổ tổ hợp nhánh tìm kiếm, giá trị của hàm lượng giá heuristic, ưu/nhược điểm của mạng nơ-ron, học tăng cường tự chơi và tìm kiếm cây dẫn hướng bằng mạng (Hybrid MCTS).

### 1.2. Mục tiêu kỹ thuật & học thuật
1. Mô hình hóa chính xác trò chơi thành môi trường ra quyết định (`state`, `legal_actions`, `transition`, `terminal`, `reward`) với giao diện API thống nhất, được kiểm thử tự động toàn diện.
2. Cài đặt đúng nguyên lý và giải thích được giả định của 7 cấp độ AI (`Random`, `Rule-based`, `Minimax/Alpha-Beta`, `MCTS UCT`, `Supervised Fast/Strong Policy Network`, `RL Q-learning / Self-play`, `Hybrid MCTS + Policy`).
3. Thực nghiệm đối kháng cân bằng lượt đi trước/sau, kiểm soát công bằng ngân sách tính toán (không chỉ cố định độ sâu tìm kiếm), đo lường độ trễ (`mean`, `p95`), chi phí tính toán (`nodes/simulations`, RAM/VRAM, thời gian train) và thực hiện phân tích ablation có ý nghĩa.
4. Trả lời bằng chứng thực nghiệm cho **3 câu hỏi nghiên cứu chính (RQ3, RQ4, RQ5)** thay vì chỉ xếp hạng bảng điểm các cấp AI.

### 1.3. Ba Câu hỏi Nghiên cứu Chính (Research Questions)

Dự án tập trung trả lời 3 câu hỏi nghiên cứu trọng tâm đã thống nhất:

1. **RQ3 — Hiệu quả của mô hình lai (Hybrid MCTS + Policy/Value Network) so với MCTS thuần và Policy-only:**
   > *Trên không gian bàn cờ Tic-Tac-Toe mở rộng (`4×4, K=4`), việc dẫn hướng tìm kiếm cây bằng mạng học được (Hybrid MCTS + Policy/Value Network theo hướng AlphaGo-inspired) cải thiện hiệu quả sử dụng ngân sách tìm kiếm (tỷ lệ thắng và chỉ số `score` theo từng mốc số lượng mô phỏng và độ trễ `ms/nước`) ở mức độ nào so với MCTS UCT sử dụng rollout ngẫu nhiên thuần túy và so với Policy-only không sử dụng tìm kiếm cây?*
   - **Giả thuyết kiểm chứng:** Khi kích thước bàn cờ mở rộng, rollout ngẫu nhiên của MCTS thuần mất dần tín hiệu chiến thuật ở đầu và trung cuộc; việc đưa phân phối xác suất tiên nghiệm $P(s, a)$ từ mạng Policy vào công thức chọn nhánh (PUCT/UCT dẫn hướng) giúp hội tụ vào nhánh tốt nhanh hơn với ít simulation hơn, vượt trội cả Policy-only (do có lookahead sửa sai chiến thuật) lẫn Pure MCTS cùng ngân sách mô phỏng.
   - **Thí nghiệm & Độ đo:** Ablation so sánh `Policy-only` vs. `Pure MCTS (UCT)` vs. `Hybrid MCTS + Policy` tại nhiều mức ngân sách (`n_simulations = 50, 150, 400` và cùng ngưỡng thời gian `ms/nước`).

2. **RQ4 — Lợi thế người đi trước (First-player Advantage) theo cấp độ AI và luật thắng:**
   > *Mức độ chênh lệch lợi thế người đi trước ($\Delta\text{Score} = \text{Score}_X - \text{Score}_O$) biến thiên như thế nào khi nâng cấp tác tử từ Baseline (Random, Rule-based) lên Tìm kiếm đối kháng (Alpha-Beta, MCTS) và Học máy/Học sâu (NN, RL, Hybrid MCTS), đồng thời việc thay đổi điều kiện thắng từ `4×4, K=3` sang `4×4, K=4` (đối chứng thêm với `3×3, K=3`) làm dịch chuyển cân bằng thắng/hòa ra sao?*
   - **Giả thuyết kiểm chứng:** Trên `4×4, K=3`, lợi thế đi trước (`X`) bị khuếch đại cực mạnh và dễ tạo thế tấn công đôi sớm; khi tăng điều kiện thắng lên `4×4, K=4`, không gian phòng thủ rộng hơn kéo ván đấu giữa các AI mạnh về tỷ lệ hòa cao hơn và thu hẹp $\Delta\text{Score}$, phản ánh rõ hơn năng lực chiến thuật dài hạn thay vì ưu thế nước đi đầu.
   - **Thí nghiệm & Độ đo:** Tách riêng ma trận kết quả khi cầm quân `X` (đi trước) và quân `O` (đi sau), tính $\Delta\text{Score}$ kèm khoảng tin cậy Wilson/Bootstrap 95% trên `3×3 (K=3)`, `4×4 (K=3)` và `4×4 (K=4)`.

3. **RQ5 — Độ ổn định & Khả năng tổng quát hóa: Reinforcement Learning (Q-learning / Self-play) so với Supervised Learning:**
   > *Khi đối đầu với tập đối thủ đa dạng chưa từng xuất hiện trong quá trình tạo dữ liệu huấn luyện (unseen opponents như Alpha-Beta giới hạn độ sâu và MCTS UCT), tác tử học tăng cường qua tự chơi (RL: Tabular Q-learning trên `3×3` / DQN Self-play trên `4×4`) có đạt khả năng tổng quát hóa và độ ổn định qua nhiều `random seed` tốt hơn so với mô hình Supervised Learning bắt chước nước đi chuyên gia (`Fast/Strong Policy Network` — vốn có nguy cơ học lại heuristic và bị giới hạn bởi trần năng lực của teacher) hay không?*
   - **Giả thuyết kiểm chứng:** Mô hình Supervised có thể đạt tỷ lệ khớp nhãn cao trên tập test cùng phân phối dữ liệu chuyên gia, nhưng dễ sụp đổ khi gặp các trạng thái lạ (out-of-distribution) từ đối thủ chơi khác phong cách; ngược lại, RL tự chơi khám phá đa dạng trạng thái hơn nhưng đòi hỏi kiểm soát exploration và kiến trúc xấp xỉ hàm giá trị (DQN) trên `4×4` để ổn định giữa các seed.
   - **Thí nghiệm & Độ đo:** Learning curves (loss / win-rate theo epoch/episode qua ít nhất 3 `seed`), tỷ lệ thắng/hòa/thua trước tập đối thủ unseen, phân tích các ván thua điển hình (failure modes).

---

## 2. Mô hình hóa bài toán & Luật chơi (`M×N, K`)

### 2.1. Thuộc tính môi trường ra quyết định
| Thuộc tính | Mô tả trong Extended Tic-Tac-Toe |
|---|---|
| Số người chơi | 2 người chơi đối kháng luân phiên (`Player X = +1` đi trước, `Player O = -1` đi sau) |
| Tính quan sát | Quan sát đầy đủ (*Fully Observable*) — cả hai thấy toàn bộ bàn cờ hiện tại |
| Tính xác định | Tất định (*Deterministic*) — một nước đi hợp lệ dẫn đến đúng 1 trạng thái kế tiếp |
| Tính tổng bằng không | Có (*Zero-sum*) — thưởng kết thúc $R_X + R_O = 0$ (`+1/-1` hoặc `0/0`) |
| Độ phức tạp không gian | `3×3`: tối đa $3^9 = 19,683$ cấu hình ($\le 9!$ chuỗi đi); `4×4`: tối đa $3^{16} \approx 4.3 \times 10^7$ cấu hình ($\le 16! \approx 2.09 \times 10^{13}$ chuỗi đi) |

### 2.2. Các cấu hình bàn cờ hỗ trợ
Engine được viết tổng quát cho mọi bộ tham số `(rows=M, cols=N, win_k=K)`:
1. **`4×4, K=4` (Primary Benchmark):** Bài toán chính dùng cho đánh giá toàn bộ 7 cấp AI, giải đấu Round-Robin và trả lời RQ3, RQ4, RQ5.
2. **`4×4, K=3` (RQ4 Sensitivity Variant):** Biến thể phục vụ riêng thực nghiệm đo độ nhạy lợi thế người đi trước theo điều kiện thắng `K`.
3. **`3×3, K=3` (Slide Sanity Baseline):** Chế độ kiểm chứng nhanh và đối chứng trực tiếp với số liệu trong `Tic tac toe Game.pdf` (Newell & Simon, `MiniMax_ab`, `MCTS`, `Fast/Strong Policy`, `Tabular Q-learning`).
4. **`5×5, K=4` (Optional Generalization Bonus):** Thử nghiệm mở rộng tính tổng quát hóa của kiến trúc Engine/Agents để phục vụ phần điểm khuyến khích.

### 2.3. Định nghĩa State, Action, Transition, Reward & API chuẩn
- **State (`s`):** Bàn cờ kích thước `(M, N)` với giá trị `{+1 (X), -1 (O), 0 (Trống)}` và `current_player ∈ {+1, -1}`. Khi đưa vào mạng nơ-ron (Cấp 5–7), bàn cờ được chuẩn hóa theo góc nhìn người chơi hiện tại (*canonical board*: `board * current_player`, shape `(M, N, 1)` theo đúng thiết kế trong slide 38).
- **Action (`a`):** Số nguyên `a ∈ {0, 1, ..., M*N - 1}` ánh xạ theo hàng-cột `(r, c) = divmod(a, N)`. Chỉ chấp nhận các ô đang trống (`board[r, c] == 0`).
- **Transition (`T(s, a)`):** Đặt quân của `current_player` vào ô `(r, c)`, cập nhật lịch sử nước đi, kiểm tra điều kiện thắng quanh nước vừa đặt theo 4 trục (ngang, dọc, chéo chính, chéo phụ) với ngưỡng `K` quân liên tiếp, đổi lượt nếu chưa kết thúc.
- **Terminal & Reward:**
  - Thắng: `winner ∈ {+1, -1}` (có ít nhất `K` quân liên tiếp) $\rightarrow$ `result(player)` trả về `+1.0` cho người thắng, `-1.0` cho người thua.
  - Hòa: Bàn cờ đầy ($M \times N$ nước) mà chưa có người thắng $\rightarrow$ `0.0` cho cả hai.
  - Chưa kết thúc: `None` / reward tức thời `0.0`.
- **Hợp đồng API Game thống nhất (bắt buộc theo Đề cương mục 3.1):**
  - `reset() -> State`
  - `legal_actions() -> list[int]`
  - `step(action: int) -> tuple[State, float, bool, dict]` (cấm nước đi bất hợp lệ bằng exception tường minh `IllegalMoveError`)
  - `is_terminal() -> bool`
  - `result(player: int | None = None) -> float | None`
  - `render() -> str`
  - `clone() -> Game` và `undo() -> None`

---

## 3. Đặc tả 7 Cấp độ Tác tử AI (Gói 2)

| Cấp | Tên tác tử | Nhóm phương pháp | Mô tả kỹ thuật cốt lõi |
|---|---|---|---|
| **1** | `RandomAgent` | Baseline | Chọn đều ngẫu nhiên 1 hành động từ `legal_actions()` với seed cố định. |
| **2** | `RuleBasedAgent` | Heuristic / Luật | Tổng quát hóa bộ luật Newell & Simon cho `M×N, K` theo thứ tự ưu tiên nghiêm ngặt (mô tả bên dưới). |
| **3** | `MinimaxAgent` / `AlphaBetaAgent` | Adversarial Search | Tìm kiếm đối kháng giới hạn độ sâu (`max_depth`) + hàm lượng giá heuristic + cắt tỉa Alpha-Beta (có cờ `use_alpha_beta` để ablation) + Iterative Deepening khi chạy theo `time_budget_ms`. |
| **4** | `MCTSAgent` | Tree Search (UCT) | 4 pha Selection (UCT với hằng số `c_uct`), Expansion, Simulation (rollout ngẫu nhiên), Backpropagation; hỗ trợ chạy theo `n_simulations` hoặc `time_budget_ms`. |
| **5** | `PolicyNetworkAgent` / `ValueGreedyAgent` | Deep Learning (Supervised) | Gồm **`FastPolicyNet`**, **`StrongPolicyNet`** (`Conv2d 3×3, 128 filters` + các lớp `Dense(64)` + `Dense(M*N)` Softmax) và **`ValueNet`** (`Tanh` đầu ra $v \in [-1, +1]$ để đối chiếu slide 61–62). Train từ dữ liệu expert tách theo `game_id`; luôn mask xác suất ô không hợp lệ khi suy luận. |
| **6** | `RLAgent` | Reinforcement Learning | Gồm **`TabularQLearningAgent`** (dùng canonical state cho `3×3, K=3` để đối chứng slide 56) và **`DQNAgent` Self-Play** (mạng Q xấp xỉ cho `4×4` và `M×N`, dùng replay buffer, target network, $\epsilon$-greedy decay) để tránh bùng nổ trạng thái trên bàn lớn. |
| **7** | `HybridMCTSAgent` | AlphaGo-inspired | Kết hợp cây tìm kiếm MCTS (PUCT) với `StrongPolicyNet` cung cấp prior $P(s, a)$ và hỗ trợ 2 chế độ đánh giá lá (`rollout` thuần và `mixed` kết hợp `ValueNet` theo hệ số $\lambda$) để trả lời trực tiếp **RQ3**. |

### 3.1. Chi tiết xử lý xung đột luật ở Cấp 2 (`RuleBasedAgent` trên `M×N, K`)
Thứ tự ưu tiên từ cao xuống thấp (nếu thỏa mãn nhiều điều kiện cùng lúc sẽ lấy điều kiện có độ ưu tiên cao hơn; đồng hạng sẽ ưu tiên ô có trọng số chiến lược cao hơn hoặc chọn ngẫu nhiên có kiểm soát):
1. **Thắng ngay (Immediate Win):** Ô trống giúp bản thân đạt đủ `K` quân liên tiếp ngay lập tức.
2. **Chặn thua ngay (Immediate Block):** Ô trống ngăn đối phương hoàn thành `K` quân liên tiếp ở lượt kế tiếp.
3. **Tạo thế đôi (Create Fork):** Ô trống tạo ra đồng thời $\ge 2$ đường mở có ít nhất $K-1$ quân mình và chưa bị đối phương chặn.
4. **Chặn thế đôi đối phương (Block Opponent Fork):** Ngăn ô cho phép đối phương tạo thế đôi; nếu đối phương có nhiều ô fork tiềm năng, ưu tiên nước vừa ép đối phương phải đỡ (tạo chuỗi $K-1$ đe dọa) vừa kiểm soát đường chéo/trung tâm.
5. **Kiểm soát ô chiến lược:** Ưu tiên vùng trung tâm $\rightarrow$ góc $\rightarrow$ cạnh theo ma trận trọng số số lượng đường thắng $K$ đi qua ô đó.
6. **Dự phòng:** Chọn ngẫu nhiên trong các nước hợp lệ còn lại.

### 3.2. Minh bạch phạm vi Cấp 7 (`AlphaGo-inspired`)
Tuân thủ lưu ý thuật ngữ của đề cương mục 3.2, `HybridMCTSAgent` được định nghĩa là **mô hình mô phỏng cảm hứng từ AlphaGo** chứ không tuyên bố tái hiện toàn bộ hệ thống AlphaGo nguyên bản:
- **Thành phần đã hiện thực:** Supervised Policy Network (`FastPolicyNet`/`StrongPolicyNet`) và `ValueNet` huấn luyện trên dữ liệu chuyên gia; tích hợp prior $P(s, a)$ vào công thức PUCT của MCTS; đánh giá nút lá theo 2 chế độ `rollout` và `mixed` ($\lambda V_\theta(s) + (1-\lambda)z_{\text{rollout}}$).
- **Thành phần chưa hiện thực:** Mạng ResNet quy mô lớn hàng chục block, kiến trúc phân tán bất đồng bộ trên nhiều GPU/TPU, và vòng lặp AlphaZero hoàn chỉnh (Cấp 8 — để dành cho hướng phát triển hoặc mở rộng sau nếu còn thời gian).

---

## 4. Thiết kế Huấn luyện, Giải đấu & Thí nghiệm Ablation

### 4.1. Quy trình dữ liệu & Huấn luyện (Cấp 5 & Cấp 6)
- **Supervised Learning (Cấp 5):**
  - Nguồn dữ liệu: Sinh từ tập các ván đấu giữa `AlphaBetaAgent` (độ sâu cao + nhiễu nhẹ ở đầu ván để tăng đa dạng trạng thái) và `RuleBasedAgent` / `MCTSAgent`.
  - Quy mô đề xuất: Tối thiểu `5,000–15,000` ván đấu hợp lệ cho `4×4, K=4`, ghi nhận `(canonical_board, action_label, game_id, outcome)`.
  - Chống rò rỉ: Loại trùng lặp cấu hình nếu cần (có ghi rõ tỷ lệ trùng trước/sau lọc) và bắt buộc chia **`train / validation / test` (70% / 15% / 15%) theo `game_id`**.
  - Lưu checkpoint tốt nhất theo `val_loss` / `val_accuracy`, ghi log epoch, seed, thời gian train và tài nguyên phần cứng.
- **Reinforcement Learning (Cấp 6):**
  - `3×3, K=3`: `TabularQLearningAgent` huấn luyện tối thiểu `50,000–100,000` episodes tự chơi + đấu đối thủ ngẫu nhiên/bán ngẫu nhiên.
  - `4×4, K=4` (và `K=3`): `DQNAgent` tự chơi (`self-play` xen kẽ đánh với pool đối thủ baseline) tối thiểu `20,000–50,000` episodes, chạy lặp lại với ít nhất **3 `random seed`** để vẽ đường cong học tập (`mean ± std` win-rate & loss).

### 4.2. Giải đấu Round-Robin Công bằng
- **Số ván đấu:** Tối thiểu **100 ván/cặp** (khuyến nghị chuẩn **200 ván/cặp**: 100 ván Agent A đi `X`, 100 ván Agent A đi `O`).
- **2 chế độ đo độc lập:**
  1. **Chế độ A — Ngân sách tính toán cố định (*Fixed Time / Compute Budget*):** Ép các tác tử tìm kiếm (`AlphaBeta`, `MCTS`, `Hybrid MCTS`) chạy trong cùng giới hạn thời gian mỗi nước (ví dụ `10 ms/move` và `50 ms/move`) để so sánh công bằng với tốc độ suy luận thực tế của `RuleBased`, `PolicyNet`, `RLAgent`.
  2. **Chế độ B — Cấu hình mặc định minh bạch (*Default Config*):** Báo cáo thông số cấu hình đầy đủ (ví dụ `AlphaBeta d=4`, `MCTS 200 rollouts`, `Hybrid MCTS 200 sims`).
- **Chỉ số cốt lõi (theo Đề cương mục 3.5):**
  - Tỷ lệ Thắng / Hòa / Thua (`W / D / L`) tách riêng quân `X`, quân `O` và tổng hợp.
  - `Score = (Thắng + 0.5 × Hòa) / Tổng số ván`, kèm **khoảng tin cậy 95% (Wilson score interval / Bootstrap CI)**.
  - Độ trễ nước đi: `mean_ms`, `p95_ms`, `max_ms`.
  - Chi phí tìm kiếm & bộ nhớ: `avg_nodes_visited` / `avg_simulations`, RAM/VRAM đỉnh, thời gian huấn luyện.

### 4.3. Kế hoạch Ablation & Sensitivity Analysis (Bắt buộc $\ge 2$, dự kiến 4 thí nghiệm)
1. **Ablation 1 (Tìm kiếm Alpha-Beta — Theo đề cương):** Bật vs. Tắt cắt tỉa Alpha-Beta và bật vs. tắt sắp xếp nước đi (*Move Ordering*) ở cùng độ sâu trên `4×4, K=4` (đo số nút duyệt và `ms/nước`).
2. **Ablation 2 (Trả lời RQ3 — Policy vs. Pure MCTS vs. Hybrid MCTS):** So sánh `FastPolicyNet`, `StrongPolicyNet`, `Pure MCTS (UCT)`, và `Hybrid MCTS + Policy` tại các ngưỡng ngân sách `n_simulations ∈ {50, 150, 400}` và ngưỡng thời gian `10 ms / 50 ms`.
3. **Ablation 3 (Trả lời RQ4 — First-player Advantage theo `K` & cấp AI):** Đối chiếu ma trận thắng `X` vs. `O` và $\Delta\text{Score}$ trên `3×3 (K=3)`, `4×4 (K=3)`, và `4×4 (K=4)`.
4. **Ablation 4 (Trả lời RQ5 — RL Self-Play vs. Supervised Learning):** So sánh `PolicyNetwork` (Supervised) vs. `RLAgent` (Self-play) qua 3 `seed` khi đối đầu với tập đối thủ unseen (`AlphaBeta`, `MCTS`).

---

## 5. Yêu cầu Giao diện Demo (`Rich CLI`) & Trực quan hóa

### 5.1. Yêu cầu Giao diện `Rich CLI`
- Hỗ trợ chọn cấu hình bàn cờ (`3×3 K=3`, `4×4 K=4`, `4×4 K=3`, tùy chỉnh `M×N, K`).
- Hỗ trợ 2 chế độ chơi: **Người–Máy (`Human vs AI`)** (cho phép người chọn cầm `X` hoặc `O`) và **Máy–Máy (`AI vs AI`)** (có thể điều chỉnh tốc độ step-by-step hoặc auto-play).
- Bảng điều khiển trực quan trên Terminal hiển thị theo thời gian thực ở mỗi nước đi:
  - Bàn cờ màu `M×N` với tọa độ rõ ràng (`r, c` hoặc `1..MN`), tô màu quân `X`, `O` và làm nổi bật nước vừa đánh.
  - **Chỉ số suy luận (Explainability Panel):** Thời gian suy nghĩ (`ms`), số nút/simulations đã duyệt, độ sâu đạt được (với Alpha-Beta), và bảng **Top-K nước đi ứng viên / Heatmap xác suất hoặc Q-value / số lượt thăm MCTS** dạng ASCII/màu trực tiếp trên terminal.
  - Kết quả cuối ván rõ ràng (tô sáng đường thắng `K` quân liên tiếp nếu có).

### 5.2. Trực quan hóa & Tài liệu phục vụ báo cáo
Pipeline phân tích tự động đọc raw JSONL logs và sinh:
- Ma trận nhiệt (*Heatmaps*) tỷ lệ thắng & `Score` tách riêng khi đi `X` và đi `O` (tương tự slide 52, 61).
- Biểu đồ độ trễ `mean_ms` & `p95_ms` (thang log) và số nút duyệt trung bình (tương tự slide 53, 62).
- Đường cong học tập (*Learning Curves*) của Supervised (loss/accuracy) và RL (reward/win-rate theo episode qua nhiều seed).
- Biểu đồ phân tích Ablation cho RQ3, RQ4, RQ5.

---

## 6. Tiêu chí Nghiệm thu & Ánh xạ Rubric Đánh giá

| STT | Tiêu chí Rubric (`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`) | Điểm tối đa | Tiêu chí nghiệm thu của hệ thống |
|---|---|---|---|
| 1 | Xác định bài toán / game | 10 | Luật `M×N, K`, mô hình MDP đầy đủ, 3 RQ rõ ràng, bộ `pytest` kiểm thử engine đạt 100%. |
| 2 | Xây dựng hệ thống | 15 | Kiến trúc tách lớp, 7 API chuẩn, `clone()/undo()` chính xác, cấm nước bất hợp lệ, Rich CLI mượt. |
| 3 | Tác tử tìm kiếm (Cấp 1–4) | 20 | `Random`, `Rule-based` (có thứ tự & xử lý xung đột), `Minimax/Alpha-Beta` (so sánh bật/tắt cắt tỉa), `MCTS UCT` đúng chuẩn 4 pha. |
| 4 | Tác tử học được (Cấp 5–7) | 15 | `Fast/Strong PolicyNet` và `RL (Tabular Q / DQN)` được huấn luyện thực sự; có dataset split theo ván, learning curves, checkpoint, và tích hợp `Hybrid MCTS`. |
| 5 | Thiết kế đánh giá | 15 | Round-robin 100–200 ván/cặp, cân bằng lượt `X/O`, đa `seed`, đo theo ngân sách thời gian/mô phỏng công bằng, $\ge 2$ ablations có ý nghĩa. |
| 6 | Kết quả & thảo luận | 10 | Ma trận đối đầu, khoảng tin cậy 95%, độ trễ p95/chi phí, trả lời RQ3/RQ4/RQ5 có căn cứ, phân tích lỗi & hạn chế. |
| 7 | Báo cáo, slide, demo | 10 | Biểu đồ tự động từ log thật, demo CLI ổn định ở cả chế độ Người–Máy và Máy–Máy, tài liệu đầy đủ. |
| 8 | Tái lập & đạo đức | 5 | Code PEP 8, README tái lập 1-lệnh, lưu raw log & seed, trích dẫn phần kế thừa minh bạch. |
| + | Điểm khuyến khích (tùy chọn) | +10 | Thực nghiệm thuyết phục `Hybrid MCTS` (AlphaGo-inspired), thử nghiệm đa cấu hình bàn (`3×3`, `4×4 K=3/4`, `5×5 K=4`), giải thích nước đi trên Rich CLI. |
