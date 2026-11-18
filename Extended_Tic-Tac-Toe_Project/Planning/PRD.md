# 01. Định hướng Đề tài, Câu hỏi Nghiên cứu & Mô hình Bài toán (`PRD.md`)

- **Vai trò tài liệu:** Sổ tay xác định mục tiêu, câu hỏi nghiên cứu và phạm vi kỹ thuật (Tài liệu số `01` trong bộ 6 tài liệu `Planning/`)
- **Môn học:** Trí tuệ nhân tạo nâng cao (Advanced AI)
- **Tài liệu tham chiếu nền tảng:**
  - Đề cương môn học: `TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`
  - Slide thiết kế thuật toán: `Tic tac toe Game.pdf`
  - Sách tham khảo chính: Mark Liu (2024), *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games*

> **Lộ trình đọc bộ tài liệu `Planning/` (01 → 06):**
> Để theo dõi mạch tư duy từ ý tưởng đến kết quả thực nghiệm (do tên file giữ chuẩn kỹ thuật quốc tế), nên đọc theo thứ tự sau:
>
> | Thứ tự | Tên file | Vai trò trong đồ án |
> |---|---|---|
> | **01** | [`PRD.md`](./PRD.md) *(đang xem)* | Mục tiêu đề tài, 3 câu hỏi nghiên cứu (`RQ1–RQ3`), mô hình MDP `M×N, K`, 7 cấp AI & Rubric |
> | **02** | [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Sổ tay kiến trúc phân lớp, luồng dữ liệu, schema log JSONL & checkpoint metadata |
> | **03** | [`DESIGN.md`](./DESIGN.md) | Thiết kế giao diện `Rich CLI` Dashboard, quy chuẩn 5 biểu đồ (`Fig 1..5`) & UX |
> | **04** | [`RULES.md`](./RULES.md) | Quy ước kỹ thuật, chuẩn code PEP 8 và nguyên tắc trung thực học thuật |
> | **05** | [`TASKS.md`](./TASKS.md) | Kế hoạch chia 5 Sprint và đầu việc kiểm chứng cho từng giai đoạn |
> | **06** | [`PROGRESS.md`](./PROGRESS.md) | Nhật ký thực nghiệm, bảng số liệu chính thức, lịch sử gỡ lỗi (`BUG-01..04`) |

---

## 1. Tổng quan & Mục tiêu Nghiên cứu

### 1.1. Lý do chọn bài toán Extended Tic-Tac-Toe (`M×N, K`)
Mục tiêu phần này là xác định phạm vi bài toán đủ phong phú để phân tách rõ năng lực giữa các nhóm phương pháp AI.

Trên bàn cờ `3×3` truyền thống, không gian trạng thái nhỏ ($3^9 = 19,683$ cấu hình) khiến các tác tử mạnh như `RuleBasedAgent` chuẩn hay `AlphaBetaAgent` toàn cây dễ kéo nhau về thế hòa, làm mờ sự khác biệt giữa tìm kiếm cây, học có giám sát và học tăng cường. Vì vậy, hệ thống được thiết kế tổng quát theo tham số `M×N, K` và chọn **`4×4, K=4`** làm **cấu hình thực nghiệm chính** để làm bộc lộ rõ:
- Sự bùng nổ tổ hợp nhánh tìm kiếm khi kích thước bàn cờ tăng lên;
- Giá trị thực tế của hàm lượng giá heuristic và sắp xếp nước đi (*Move Ordering*);
- Khác biệt về khả năng tổng quát hóa giữa Supervised Learning, Reinforcement Learning và tìm kiếm cây dẫn hướng bằng mạng (Hybrid MCTS).

### 1.2. Mục tiêu kỹ thuật & học thuật
1. Mô hình hóa chính xác trò chơi thành môi trường ra quyết định đối kháng (`state`, `legal_actions`, `transition`, `terminal`, `reward`) với giao diện API thống nhất và bộ kiểm thử tự động đầy đủ.
2. Cài đặt đúng nguyên lý và làm rõ giả định hoạt động của 7 cấp độ tác tử AI (`Random`, `Rule-based`, `Minimax/Alpha-Beta`, `MCTS UCT`, `Supervised Fast/Strong Policy & Value Network`, `RL Q-learning / DQN Self-play`, `Hybrid MCTS + Policy/Value`).
3. Thiết kế thực nghiệm đối kháng cân bằng lượt đi trước/sau (`X` và `O`), kiểm soát công bằng ngân sách tính toán (theo cả độ sâu/số mô phỏng lẫn giới hạn thời gian `ms/nước`), đo độ trễ (`mean`, `p95`), chi phí tìm kiếm (`nodes/simulations`, RAM/VRAM, thời gian train) và thực hiện phân tích ablation có ý nghĩa thống kê.
4. Trả lời bằng chứng thực nghiệm cho **3 câu hỏi nghiên cứu chính (RQ1, RQ2, RQ3)** thay vì chỉ dừng lại ở bảng xếp hạng thắng/thua.

### 1.3. Ba Câu hỏi Nghiên cứu Trọng tâm (Research Questions)

1. **RQ1 — Hiệu quả của mô hình lai (Hybrid MCTS + Policy/Value Network) so với MCTS thuần và Policy-only:**
   > *Trên không gian bàn cờ Tic-Tac-Toe mở rộng (`4×4, K=4`), việc dẫn hướng tìm kiếm cây bằng mạng học được (Hybrid MCTS + Policy/Value Network theo hướng AlphaGo-inspired) cải thiện hiệu quả sử dụng ngân sách tìm kiếm (tỷ lệ thắng và chỉ số `score` theo từng mốc số lượng mô phỏng và độ trễ `ms/nước`) ở mức độ nào so với MCTS UCT sử dụng rollout ngẫu nhiên thuần túy và so với Policy-only không sử dụng tìm kiếm cây?*
   - **Giả thuyết kiểm chứng:** Khi kích thước bàn cờ mở rộng, rollout ngẫu nhiên của MCTS thuần mất dần tín hiệu chiến thuật ở đầu và trung cuộc; việc đưa phân phối xác suất tiên nghiệm $P(s, a)$ từ mạng Policy vào công thức chọn nhánh (PUCT/UCT dẫn hướng) giúp hội tụ vào nhánh tốt nhanh hơn với ít simulation hơn, vượt trội cả Policy-only (do có lookahead sửa sai chiến thuật) lẫn Pure MCTS cùng ngân sách mô phỏng.
   - **Thí nghiệm & Độ đo:** Ablation so sánh `Policy-only` vs. `Pure MCTS (UCT)` vs. `Hybrid MCTS + Policy` tại nhiều mức ngân sách (`n_simulations = 50, 150, 400` và cùng ngưỡng thời gian `ms/nước`).

2. **RQ2 — Lợi thế người đi trước (First-player Advantage) theo cấp độ AI và luật thắng:**
   > *Mức độ chênh lệch lợi thế người đi trước ($\Delta\text{Score} = \text{Score}_X - \text{Score}_O$) biến thiên như thế nào khi nâng cấp tác tử từ Baseline (Random, Rule-based) lên Tìm kiếm đối kháng (Alpha-Beta, MCTS) và Học máy/Học sâu (NN, RL, Hybrid MCTS), đồng thời việc thay đổi điều kiện thắng từ `4×4, K=3` sang `4×4, K=4` (đối chứng thêm với `3×3, K=3`) làm dịch chuyển cân bằng thắng/hòa ra sao?*
   - **Giả thuyết kiểm chứng:** Trên `4×4, K=3`, lợi thế đi trước (`X`) bị khuếch đại cực mạnh và dễ tạo thế tấn công đôi sớm; khi tăng điều kiện thắng lên `4×4, K=4`, không gian phòng thủ rộng hơn kéo ván đấu giữa các AI mạnh về tỷ lệ hòa cao hơn và thu hẹp $\Delta\text{Score}$, phản ánh rõ hơn năng lực chiến thuật dài hạn thay vì ưu thế nước đi đầu.
   - **Thí nghiệm & Độ đo:** Tách riêng ma trận kết quả khi cầm quân `X` (đi trước) và quân `O` (đi sau), tính $\Delta\text{Score}$ kèm khoảng tin cậy Wilson/Bootstrap 95% trên `3×3 (K=3)`, `4×4 (K=3)` và `4×4 (K=4)`.

3. **RQ3 — Độ ổn định & Khả năng tổng quát hóa: Reinforcement Learning (Q-learning / Self-play) so với Supervised Learning:**
   > *Khi đối đầu với tập đối thủ đa dạng chưa từng xuất hiện trong quá trình tạo dữ liệu huấn luyện (unseen opponents như Alpha-Beta giới hạn độ sâu và MCTS UCT), tác tử học tăng cường qua tự chơi (RL: Tabular Q-learning trên `3×3` / DQN Self-play trên `4×4`) có đạt khả năng tổng quát hóa và độ ổn định qua nhiều `random seed` tốt hơn so với mô hình Supervised Learning bắt chước nước đi chuyên gia (`Fast/Strong Policy Network` — vốn có nguy cơ học lại heuristic và bị giới hạn bởi trần năng lực của teacher) hay không?*
   - **Giả thuyết kiểm chứng:** Mô hình Supervised có thể đạt tỷ lệ khớp nhãn cao trên tập test cùng phân phối dữ liệu chuyên gia, nhưng dễ sụp đổ khi gặp các trạng thái lạ (out-of-distribution) từ đối thủ chơi khác phong cách; ngược lại, RL tự chơi khám phá đa dạng trạng thái hơn nhưng đòi hỏi kiểm soát exploration và kiến trúc xấp xỉ hàm giá trị (DQN) trên `4×4` để ổn định giữa các seed.
   - **Thí nghiệm & Độ đo:** Learning curves (loss / win-rate theo epoch/episode qua ít nhất 3 `seed`), tỷ lệ thắng/hòa/thua trước tập đối thủ unseen, phân tích các ván thua điển hình (failure modes).

---

## 2. Mô hình hóa Bài toán & Luật chơi (`M×N, K`)

### 2.1. Thuộc tính môi trường ra quyết định
| Thuộc tính | Mô tả trong Extended Tic-Tac-Toe |
|---|---|
| Số người chơi | 2 người chơi đối kháng luân phiên (`Player X = +1` đi trước, `Player O = -1` đi sau) |
| Tính quan sát | Quan sát đầy đủ (*Fully Observable*) — cả hai thấy toàn bộ bàn cờ hiện tại |
| Tính xác định | Tất định (*Deterministic*) — một nước đi hợp lệ dẫn đến đúng 1 trạng thái kế tiếp |
| Tính tổng bằng không | Có (*Zero-sum*) — thưởng kết thúc $R_X + R_O = 0$ (`+1/-1` hoặc `0/0`) |
| Độ phức tạp không gian | `3×3`: tối đa $3^9 = 19,683$ cấu hình ($\le 9!$ chuỗi đi); `4×4`: tối đa $3^{16} \approx 4.3 \times 10^7$ cấu hình ($\le 16! \approx 2.09 \times 10^{13}$ chuỗi đi) |

### 2.2. Các cấu hình bàn cờ sử dụng
Engine được thiết kế tổng quát cho mọi bộ tham số `(rows=M, cols=N, win_k=K)`:
1. **`4×4, K=4` (Primary Benchmark):** Cấu hình chuẩn dùng để đánh giá toàn bộ 7 cấp AI, chạy giải đấu Round-Robin và trả lời RQ1, RQ2, RQ3.
2. **`4×4, K=3` (RQ2 Sensitivity Variant):** Biến thể phục vụ thực nghiệm đo độ nhạy lợi thế người đi trước theo điều kiện thắng `K`.
3. **`3×3, K=3` (Slide Sanity Baseline):** Chế độ kiểm chứng nhanh và đối chứng trực tiếp với thiết kế trong `Tic tac toe Game.pdf` (Newell & Simon, `MiniMax_ab`, `MCTS`, `Fast/Strong Policy`, `Tabular Q-learning`).
4. **`5×5, K=4` (Generalization Bonus):** Cấu hình thử nghiệm mở rộng để kiểm chứng khả năng tổng quát hóa của kiến trúc Engine và các tác tử trên bàn cờ lớn hơn.

### 2.3. Định nghĩa State, Action, Transition, Reward & API chuẩn
- **State (`s`):** Bàn cờ kích thước `(M, N)` với giá trị `{+1 (X), -1 (O), 0 (Trống)}` và `current_player ∈ {+1, -1}`. Khi đưa vào mạng nơ-ron (Cấp 5–7), bàn cờ được chuẩn hóa theo góc nhìn người chơi hiện tại (*canonical board*: `board * current_player`, shape `(M, N, 1)` theo thiết kế trong slide 38).
- **Action (`a`):** Số nguyên `a ∈ {0, 1, ..., M*N - 1}` ánh xạ theo hàng-cột `(r, c) = divmod(a, N)`. Chỉ các ô đang trống (`board[r, c] == 0`) mới hợp lệ.
- **Transition (`T(s, a)`):** Đặt quân của `current_player` vào ô `(r, c)`, cập nhật lịch sử nước đi, kiểm tra điều kiện thắng quanh nước vừa đặt theo 4 trục (ngang, dọc, chéo chính, chéo phụ) với ngưỡng `K` quân liên tiếp, đổi lượt nếu ván đấu chưa kết thúc.
- **Terminal & Reward:**
  - Thắng: `winner ∈ {+1, -1}` (đạt ít nhất `K` quân liên tiếp) $\rightarrow$ `result(player)` trả về `+1.0` cho người thắng, `-1.0` cho người thua.
  - Hòa: Bàn cờ đầy ($M \times N$ nước) mà chưa có người thắng $\rightarrow$ `0.0` cho cả hai.
  - Chưa kết thúc: `None` / reward tức thời `0.0`.
- **Giao diện Game API thống nhất (theo Đề cương mục 3.1):**
  - `reset() -> State`
  - `legal_actions() -> list[int]`
  - `step(action: int) -> tuple[State, float, bool, dict]` (kiểm tra tính hợp lệ và ném `IllegalMoveError` khi gặp nước đi sai)
  - `is_terminal() -> bool`
  - `result(player: int | None = None) -> float | None`
  - `render() -> str`
  - `clone() -> Game` và `undo() -> None`

> **Ghi chú kinh nghiệm (Engine & Canonical State):**
> - Biểu diễn `canonical_board = board * current_player` giúp mạng nơ-ron và bảng Q chỉ cần học từ góc nhìn *"quân ta = +1, quân địch = -1"*, giảm một nửa không gian học so với việc phải phân biệt lúc cầm `X` hay `O`.
> - Khi viết các tác tử tìm kiếm cây, luôn thao tác trên bản sao `game.clone()` hoặc cặp `step()/undo()` cân bằng để bàn cờ gốc của ván đấu không bị biến đổi ngoài ý muốn.

---

## 3. Thiết kế 7 Cấp độ Tác tử AI

| Cấp | Tên tác tử | Nhóm phương pháp | Mô tả kỹ thuật cốt lõi |
|---|---|---|---|
| **1** | `RandomAgent` | Baseline | Chọn đều ngẫu nhiên 1 hành động từ `legal_actions()` với seed cố định. |
| **2** | `RuleBasedAgent` | Heuristic / Luật | Tổng quát hóa bộ luật Newell & Simon cho `M×N, K` theo thứ tự ưu tiên phân tầng rõ ràng (mô tả bên dưới). |
| **3** | `MinimaxAgent` / `AlphaBetaAgent` | Adversarial Search | Tìm kiếm đối kháng giới hạn độ sâu (`max_depth`) + hàm lượng giá heuristic + cắt tỉa Alpha-Beta (có cờ `use_alpha_beta` để ablation) + Iterative Deepening khi chạy theo `time_budget_ms`. |
| **4** | `MCTSAgent` | Tree Search (UCT) | 4 pha Selection (UCT với hằng số `c_uct`), Expansion, Simulation (rollout ngẫu nhiên), Backpropagation; hỗ trợ chạy theo `n_simulations` hoặc `time_budget_ms`. |
| **5** | `PolicyNetworkAgent` / `ValueGreedyAgent` | Deep Learning (Supervised) | Gồm **`FastPolicyNet`**, **`StrongPolicyNet`** (`Conv2d 3×3, 128 filters` + các lớp `Dense(64)` + `Dense(M*N)` Softmax) và **`ValueNet`** (`Tanh` đầu ra $v \in [-1, +1]$ để đối chiếu slide 61–62). Huấn luyện từ dữ liệu expert tách theo `game_id`; luôn mask xác suất ô không hợp lệ khi suy luận. |
| **6** | `RLAgent` | Reinforcement Learning | Gồm **`TabularQLearningAgent`** (dùng canonical state cho `3×3, K=3` để đối chứng slide 56) và **`DQNAgent` Self-Play** (mạng Q xấp xỉ cho `4×4` và `M×N`, dùng replay buffer, target network, $\epsilon$-greedy decay) để tránh bùng nổ trạng thái trên bàn lớn. |
| **7** | `HybridMCTSAgent` | AlphaGo-inspired | Kết hợp cây tìm kiếm MCTS (PUCT) với `StrongPolicyNet` cung cấp prior $P(s, a)$ và hỗ trợ 2 chế độ đánh giá lá (`rollout` thuần và `mixed` kết hợp `ValueNet` theo hệ số $\lambda$) để trả lời trực tiếp **RQ1**. |

### 3.1. Thứ tự phân giải xung đột ở Cấp 2 (`RuleBasedAgent` trên `M×N, K`)
Thứ tự ưu tiên từ cao xuống thấp (nếu thỏa mãn nhiều điều kiện cùng lúc thì lấy điều kiện có độ ưu tiên cao hơn; khi đồng hạng sẽ ưu tiên ô có trọng số chiến lược cao hơn hoặc chọn ngẫu nhiên có kiểm soát):
1. **Thắng ngay (Immediate Win):** Ô trống giúp bản thân đạt đủ `K` quân liên tiếp ngay lập tức.
2. **Chặn thua ngay (Immediate Block):** Ô trống ngăn đối phương hoàn thành `K` quân liên tiếp ở lượt kế tiếp.
3. **Tạo thế đôi (Create Fork):** Ô trống tạo ra đồng thời $\ge 2$ đường mở có ít nhất $K-1$ quân mình và chưa bị đối phương chặn.
4. **Chặn thế đôi đối phương (Block Opponent Fork):** Ngăn ô cho phép đối phương tạo thế đôi; nếu đối phương có nhiều ô fork tiềm năng, ưu tiên nước vừa ép đối phương phải đỡ (tạo chuỗi $K-1$ đe dọa) vừa kiểm soát đường chéo/trung tâm.
5. **Kiểm soát ô chiến lược:** Ưu tiên vùng trung tâm $\rightarrow$ góc $\rightarrow$ cạnh theo ma trận trọng số số lượng đường thắng $K$ đi qua ô đó.
6. **Dự phòng:** Chọn ngẫu nhiên trong các nước hợp lệ còn lại.

### 3.2. Minh bạch phạm vi Cấp 7 (`AlphaGo-inspired`)
Theo lưu ý thuật ngữ của đề cương mục 3.2, `HybridMCTSAgent` được định vị là **mô hình mô phỏng cảm hứng từ AlphaGo** chứ không tái hiện toàn bộ hệ thống AlphaGo nguyên bản:
- **Thành phần đã hiện thực:** Supervised Policy Network (`FastPolicyNet`/`StrongPolicyNet`) và `ValueNet` huấn luyện trên dữ liệu chuyên gia; tích hợp prior $P(s, a)$ vào công thức PUCT của MCTS; đánh giá nút lá theo 2 chế độ `rollout` và `mixed` ($\lambda V_\theta(s) + (1-\lambda)z_{\text{rollout}}$).
- **Thành phần nằm ngoài phạm vi:** Mạng ResNet quy mô lớn hàng chục block, kiến trúc phân tán bất đồng bộ trên nhiều GPU/TPU, và vòng lặp AlphaZero hoàn chỉnh.

> **Ghi chú kinh nghiệm (Các cấp AI):**
> - Ở `AlphaBetaAgent`, cần giữ nguyên tính chất đối xứng toán học với `MinimaxAgent` thuần (cùng độ sâu, không cắt nhánh sai) để có thể kiểm chứng chéo bằng unit test trước khi bật Iterative Deepening.
> - Ở các tác tử dùng mạng nơ-ron (Cấp 5, 6, 7), việc gán logit `-1e9` vào các ô đã có quân (*Action Masking*) trước bước `Softmax` hoặc `argmax` giúp loại bỏ hoàn toàn rủi ro mạng đề xuất nước đi bất hợp lệ.

---

## 4. Kế hoạch Huấn luyện, Giải đấu & Thí nghiệm Ablation

### 4.1. Quy trình dữ liệu & Huấn luyện (Cấp 5 & Cấp 6)
- **Supervised Learning (Cấp 5):**
  - Nguồn dữ liệu: Sinh từ tập các ván đấu giữa `AlphaBetaAgent` (độ sâu cao + nhiễu nhẹ ở đầu ván để tăng đa dạng trạng thái) và `RuleBasedAgent` / `MCTSAgent`.
  - Quy mô thiết kế: Tối thiểu `5,000–15,000` ván đấu hợp lệ cho `4×4, K=4`, ghi nhận `(canonical_board, action_label, game_id, outcome)`.
  - Chống rò rỉ dữ liệu: Ghi nhận tỷ lệ trùng lặp nếu có lọc và chia **`train / validation / test` (70% / 15% / 15%) theo `game_id`** để các trạng thái trong cùng một ván không rơi đồng thời vào tập train và tập test.
  - Lưu checkpoint tốt nhất theo `val_loss` / `val_accuracy`, ghi log epoch, seed, thời gian train và tài nguyên phần cứng.
- **Reinforcement Learning (Cấp 6):**
  - `3×3, K=3`: `TabularQLearningAgent` huấn luyện tối thiểu `50,000–100,000` episodes tự chơi + đấu đối thủ ngẫu nhiên/bán ngẫu nhiên.
  - `4×4, K=4` (và `K=3`): `DQNAgent` tự chơi (`self-play` xen kẽ đánh với pool đối thủ baseline) tối thiểu `20,000–50,000` episodes, chạy lặp lại với ít nhất **3 `random seed`** để vẽ đường cong học tập (`mean ± std` win-rate & loss).

### 4.2. Giải đấu Round-Robin Công bằng
- **Số ván đấu:** Tối thiểu **100 ván/cặp** (chuẩn thực nghiệm **200 ván/cặp**: 100 ván Agent A đi `X`, 100 ván Agent A đi `O`).
- **2 chế độ đo độc lập:**
  1. **Chế độ A — Ngân sách tính toán cố định (*Fixed Time / Compute Budget*):** Đưa các tác tử tìm kiếm (`AlphaBeta`, `MCTS`, `Hybrid MCTS`) chạy trong cùng giới hạn thời gian mỗi nước (ví dụ `10 ms/move` và `50 ms/move`) để so sánh công bằng với tốc độ suy luận thực tế của `RuleBased`, `PolicyNet`, `RLAgent`.
  2. **Chế độ B — Cấu hình mặc định minh bạch (*Default Config*):** Báo cáo đầy đủ cấu hình tham chiếu (ví dụ `AlphaBeta d=4`, `MCTS 200 rollouts`, `Hybrid MCTS 200 sims`).
- **Chỉ số cốt lõi (theo Đề cương mục 3.5):**
  - Tỷ lệ Thắng / Hòa / Thua (`W / D / L`) tách riêng quân `X`, quân `O` và tổng hợp.
  - `Score = (Thắng + 0.5 × Hòa) / Tổng số ván`, kèm **khoảng tin cậy 95% (Wilson score interval / Bootstrap CI)**.
  - Độ trễ nước đi: `mean_ms`, `p95_ms`, `max_ms`.
  - Chi phí tìm kiếm & bộ nhớ: `avg_nodes_visited` / `avg_simulations`, RAM/VRAM đỉnh, thời gian huấn luyện.

### 4.3. Kế hoạch Ablation & Sensitivity Analysis (Dự kiến 4 thí nghiệm)
1. **Ablation 1 (Tìm kiếm Alpha-Beta — Theo đề cương):** Bật vs. Tắt cắt tỉa Alpha-Beta và bật vs. tắt sắp xếp nước đi (*Move Ordering*) ở cùng độ sâu trên `4×4, K=4` (đo số nút duyệt và `ms/nước`).
2. **Ablation 2 (Trả lời RQ1 — Policy vs. Pure MCTS vs. Hybrid MCTS):** So sánh `FastPolicyNet`, `StrongPolicyNet`, `Pure MCTS (UCT)`, và `Hybrid MCTS + Policy` tại các ngưỡng ngân sách `n_simulations ∈ {50, 150, 400}` và ngưỡng thời gian `10 ms / 50 ms`.
3. **Ablation 3 (Trả lời RQ2 — First-player Advantage theo `K` & cấp AI):** Đối chiếu ma trận thắng `X` vs. `O` và $\Delta\text{Score}$ trên `3×3 (K=3)`, `4×4 (K=3)`, và `4×4 (K=4)`.
4. **Ablation 4 (Trả lời RQ3 — RL Self-Play vs. Supervised Learning):** So sánh `PolicyNetwork` (Supervised) vs. `RLAgent` (Self-play) qua 3 `seed` khi đối đầu với tập đối thủ unseen (`AlphaBeta`, `MCTS`).

> **Ghi chú kinh nghiệm (Thiết kế thực nghiệm):**
> - So sánh AI chỉ có ý nghĩa khi tách rạch ròi 2 chiều: *cùng ngân sách số bước tìm kiếm* (`n_simulations`) để đánh giá chất lượng dẫn hướng thuật toán, và *cùng ngân sách thời gian* (`ms/move`) để đánh giá hiệu năng thực tế có tính đến chi phí forward PyTorch.

---

## 5. Định hướng Giao diện `Rich CLI` & Trực quan hóa

### 5.1. Yêu cầu Giao diện `Rich CLI`
- Hỗ trợ chọn cấu hình bàn cờ (`3×3 K=3`, `4×4 K=4`, `4×4 K=3`, tùy chỉnh `M×N, K`).
- Hỗ trợ 2 chế độ chơi: **Người–Máy (`Human vs AI`)** (cho phép chọn cầm `X` hoặc `O`) và **Máy–Máy (`AI vs AI`)** (có thể điều chỉnh tốc độ step-by-step hoặc auto-play).
- Bảng điều khiển trên Terminal hiển thị theo thời gian thực ở mỗi nước đi:
  - Bàn cờ màu `M×N` với tọa độ rõ ràng (`r, c` hoặc `0..MN-1`), tô màu quân `X`, `O` và làm nổi bật nước vừa đánh.
  - **Chỉ số suy luận (Explainability Panel):** Thời gian suy nghĩ (`ms`), số nút/simulations đã duyệt, độ sâu đạt được (với Alpha-Beta), và bảng **Top-K nước đi ứng viên / Heatmap xác suất hoặc Q-value / số lượt thăm MCTS** trực tiếp trên terminal.
  - Kết quả cuối ván rõ ràng (tô sáng đường thắng `K` quân liên tiếp nếu có).

### 5.2. Trực quan hóa & Tổng hợp Báo cáo
Pipeline phân tích đọc trực tiếp từ raw JSONL logs và xuất ra:
- Ma trận nhiệt (*Heatmaps*) tỷ lệ thắng & `Score` tách riêng khi đi `X` và đi `O` (tương tự slide 52, 61).
- Biểu đồ độ trễ `mean_ms` & `p95_ms` (thang log) và số nút duyệt trung bình (tương tự slide 53, 62).
- Đường cong học tập (*Learning Curves*) của Supervised (loss/accuracy) và RL (reward/win-rate theo episode qua nhiều seed).
- Biểu đồ phân tích Ablation cho RQ1, RQ2, RQ3.

---

## 6. Bảng Tự Đánh giá & Ánh xạ Rubric Môn học

| STT | Tiêu chí Rubric (`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`) | Điểm tối đa | Mục tiêu cần đạt của hệ thống |
|---|---|---|---|
| 1 | Xác định bài toán / game | 10 | Luật `M×N, K`, mô hình MDP đầy đủ, 3 RQ rõ ràng, bộ `pytest` kiểm thử engine đạt 100%. |
| 2 | Xây dựng hệ thống | 15 | Kiến trúc tách lớp, 7 API chuẩn, `clone()/undo()` chính xác, xử lý nước bất hợp lệ, Rich CLI mượt. |
| 3 | Tác tử tìm kiếm (Cấp 1–4) | 20 | `Random`, `Rule-based` (có thứ tự & xử lý xung đột), `Minimax/Alpha-Beta` (so sánh bật/tắt cắt tỉa), `MCTS UCT` đúng chuẩn 4 pha. |
| 4 | Tác tử học được (Cấp 5–7) | 15 | `Fast/Strong PolicyNet` và `RL (Tabular Q / DQN)` được huấn luyện thực sự; có dataset split theo ván, learning curves, checkpoint, và tích hợp `Hybrid MCTS`. |
| 5 | Thiết kế đánh giá | 15 | Round-robin 100–200 ván/cặp, cân bằng lượt `X/O`, đa `seed`, đo theo ngân sách thời gian/mô phỏng công bằng, $\ge 2$ ablations có ý nghĩa. |
| 6 | Kết quả & thảo luận | 10 | Ma trận đối đầu, khoảng tin cậy 95%, độ trễ p95/chi phí, trả lời RQ1/RQ2/RQ3 có căn cứ, phân tích lỗi & hạn chế. |
| 7 | Báo cáo, slide, demo | 10 | Biểu đồ tự động từ log thật, demo CLI ổn định ở cả chế độ Người–Máy và Máy–Máy, tài liệu đầy đủ. |
| 8 | Tái lập & đạo đức | 5 | Code PEP 8, README tái lập 1-lệnh, lưu raw log & seed, trích dẫn phần kế thừa minh bạch. |
| + | Điểm khuyến khích (tùy chọn) | +10 | Thực nghiệm thuyết phục `Hybrid MCTS` (AlphaGo-inspired), thử nghiệm đa cấu hình bàn (`3×3`, `4×4 K=3/4`, `5×5 K=4`), giải thích nước đi trên Rich CLI. |

---

## 7. Checklist Tự Kiểm tra Trước Khi Thiết kế Kiến trúc & Code

Trước khi chuyển sang [ARCHITECTURE.md](./ARCHITECTURE.md) hoặc chỉnh sửa mã nguồn cốt lõi, cần rà soát lại các câu hỏi sau:
- [x] Các cấu hình bàn cờ mới (`M×N, K`) có đều đi qua cùng một bộ API 7+ phương thức mà không phá vỡ logic thắng/thua theo 4 trục?
- [x] Mọi câu hỏi nghiên cứu (`RQ1`, `RQ2`, `RQ3`) đã gắn với chỉ số đo định lượng và thí nghiệm ablation tương ứng chưa?
- [x] Việc chia dữ liệu Supervised đã thực hiện theo `game_id` (`70/15/15`) thay vì chia ngẫu nhiên theo từng trạng thái đơn lẻ chưa?
- [x] Phạm vi của Cấp 7 (`HybridMCTSAgent`) đã được mô tả trung thực là *AlphaGo-inspired* (không phóng đại thành AlphaZero phân tán)?
- [x] Kế hoạch giải đấu đã tách riêng 2 chế độ *Fixed Budget* (`10 ms`, `50 ms`) và *Default Config* cũng như cân bằng 50/50 lượt `X/O` chưa?
