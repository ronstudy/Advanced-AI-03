# DESIGN — Thiết kế Giao diện Rich CLI & Hệ thống Trực quan hóa Báo cáo

Tài liệu này mô tả thiết kế chi tiết cho giao diện người dùng cuối dòng lệnh (**Rich CLI**) và hệ thống xuất biểu đồ phân tích khoa học (`visualizer.py`), đảm bảo đáp ứng đầy đủ yêu cầu demo trực quan và giải thích quyết định của AI trong [PRD.md](./PRD.md) và [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## 1. Nguyên tắc Thiết kế Giao diện `Rich CLI`

1. **Zero-Clutter Terminal Dashboard:** Tổ chức màn hình rõ ràng theo bố cục cố định bằng `rich.layout.Layout` và `rich.panel.Panel`, tránh cuộn trôi mất bàn cờ khi chơi nhiều lượt.
2. **Tách biệt Dữ liệu và Hiển thị:** UI chỉ đọc trạng thái công khai từ `ExtendedTicTacToe` và `AgentDecision` (không tự tính toán chiến thuật hoặc đọc biến private của engine).
3. **Khả năng Tương thích đa Terminal (Windows PowerShell / CMD / VS Code):**
   - Sử dụng bộ ký tự khung hộp chuẩn của `rich.box` (`ROUNDED` / `SQUARE`), tự động hỗ trợ fallback ASCII thuần nếu biến môi trường hoặc terminal không hỗ trợ UTF-8 đầy đủ.
   - Bảng màu có độ tương phản cao trên cả nền tối (*Dark theme*) và nền sáng (*Light theme*).
4. **Tính Giải thích Thời gian thực (*Real-time Explainability*):** Mỗi nước đi của AI luôn hiển thị đồng thời **bàn cờ sau nước đi**, **chi phí suy luận** (`ms`, `depth`, `nodes/simulations`), và **Heatmap phân phối điểm/xác suất trên toàn bàn cờ**.

---

## 2. Thiết kế Màn hình & Luồng Tương tác (`src/ttt_ai/ui/cli_app.py`)

### 2.1. Màn hình 1 — Setup Wizard (Khởi tạo trận đấu)
Khi chạy `python scripts/play.py` (hoặc hỗ trợ cờ CLI trực tiếp để vào thẳng trận khi demo), người dùng chọn nhanh 4 nhóm tham số:
1. **Cấu hình bàn cờ (`Board Preset`):**
   - `[1]` `4×4, K=4` *(Mặc định — Primary Benchmark)*
   - `[2]` `4×4, K=3` *(Biến thể nhạy lợi thế đi trước — RQ4)*
   - `[3]` `3×3, K=3` *(Đối chứng Slide)*
   - `[4]` `5×5, K=4` *(Mở rộng tổng quát hóa)*
   - `[5]` Tùy chỉnh `M × N, K`
2. **Chế độ chơi (`Game Mode`):**
   - `[1]` **Người – Máy (`Human vs AI`)**: Chọn người chơi cầm `X` (đi trước) hay `O` (đi sau).
   - `[2]` **Máy – Máy (`AI vs AI`)**: Chọn chế độ `Auto-play` (có `step_delay_ms`, mặc định `600 ms`) hoặc `Step-by-step` (nhấn `Enter` để AI đi từng nước khi thuyết trình).
3. **Chọn Tác tử AI (Cấp 1–7):** Menu hiển thị rõ tên, cấp độ, và trạng thái checkpoint sẵn có (với Cấp 5, 6, 7 chỉ cho chọn khi checkpoint đã huấn luyện tồn tại, tuyệt đối không cho chạy trọng số chưa train).
4. **Ngân sách & Seed (`Budget & Seed`):** Chọn `Default Config` hoặc `Fixed Time Budget` (ví dụ `10 ms`, `50 ms`, `200 ms`) và nhập `seed` (mặc định `42`).

### 2.2. Màn hình 2 — Live Match Dashboard (Trong trận đấu)

Bố cục màn hình chia làm **Header**, **Thân 2 cột** (Trái 48% / Phải 52%), và **Footer**:

```text
+-----------------------------------------------------------------------------------+
| Extended Tic-Tac-Toe AI Arena | Board: 4x4 (K=4) | Mode: Human (X) vs HybridMCTS  |
| Turn #6 | Current: Player X | Seed: 42 | Budget: 50 ms                            |
+-----------------------------------------+-----------------------------------------+
|              GAME BOARD                 |          AI TELEMETRY (Last Move)       |
|                                         |  Agent   : HybridMCTS (Level 7)         |
|       0     1     2     3               |  Move    : (1, 2) [cell 6]              |
|    +-----+-----+-----+-----+            |  Latency : 18.42 ms                     |
|  0 |  X  |     |  O  |     |            |  Search  : 200 sims | Depth: -          |
|    +-----+-----+-----+-----+            |  Value   : +0.34 (est.)                 |
|  1 |     |  X  | [O] |     |  <- Last   +-----------------------------------------+
|    +-----+-----+-----+-----+            |       ACTION HEATMAP & TOP-5 MOVES      |
|  2 |     |  O  |     |     |            |     0      1      2      3              |
|    +-----+-----+-----+-----+            |  0  .     14%     .     08%             |
|  3 |     |     |     |  X  |            |  1 05%     .    [42%]   03%             |
|    +-----+-----+-----+-----+            |  2 02%     .     19%    04%             |
|                                         |  3 01%    01%    01%     .              |
|  Legend: X=Cyan | O=Red | [ ]=Last Move |  Top: (1,2) 42% | (2,2) 19% | (0,1) 14% |
+-----------------------------------------+-----------------------------------------+
| History: 1.X(1,1) -> 2.O(0,2) -> 3.X(0,0) -> 4.O(2,1) -> 5.X(3,3) -> 6.O(1,2)     |
| Input move as "r,c" or cell index [0-15] (commands: 'u'=undo, 'h'=help, 'q'=quit): |
+-----------------------------------------------------------------------------------+
```

#### Quy ước màu sắc & hiển thị chi tiết (`src/ttt_ai/ui/theme.py`)
1. **Board Panel:**
   - Quân `X`: Màu `bold bright_cyan`.
   - Quân `O`: Màu `bold bright_red`.
   - Nước vừa đánh (*Last Move*): Viền hoặc nền highlight `on grey23` kèm ký hiệu đánh dấu.
   - Đường thắng (*Winning Line*): Khi `is_terminal()` và có người thắng, toàn bộ `K` ô tạo thành đường thắng được tô nền `bold black on bright_green` để người xem nhận diện ngay lập tức.
   - Ô trống: Hiển thị mờ chỉ số ô (`dim grey50`) hoặc để trống sạch sẽ tùy độ rộng màn hình (tự động co giãn theo `M×N`).
2. **AI Telemetry Panel:**
   - Hiển thị trực tiếp từ `AgentDecision`:
     - `Latency (ms)` (tô xanh nếu trong ngân sách, vàng nếu > 80% ngân sách, đỏ nếu vượt nhẹ).
     - `Nodes Visited` (với Minimax/Alpha-Beta) hoặc `Simulations` (với MCTS/HybridMCTS).
     - `Depth Reached` (với Minimax/Alpha-Beta Iterative Deepening).
     - `Value Estimate` (nếu có từ search tree root Q-value hoặc DQN max Q).
3. **Action Heatmap & Top-K Table Panel:**
   - Chuyển đổi `AgentDecision.action_scores` thành thang nhiệt 5 mức cho các ô hợp lệ:
     - Top 1 (nước chọn / cao nhất): `bold black on bright_yellow` (hoặc `on green`) kèm nhãn `[xx%]` hoặc `[score]`.
     - Mức cao ($\ge 60\%$ giá trị max): `bold green` / `on dark_green`.
     - Mức trung bình ($25\% - 60\%$): `yellow`.
     - Mức thấp ($< 25\%$): `dim white`.
     - Ô đã có quân: Hiển thị `.` hoặc ký hiệu mờ `X/O`.
   - **Chuẩn hóa hiển thị theo từng loại Agent:**
     - *Cấp 1 (`Random`):* Phân phối đều `1 / |legal|` trên các ô trống.
     - *Cấp 2 (`RuleBased`):* Hiển thị điểm ưu tiên tầng luật / trọng số chiến lược.
     - *Cấp 3 (`Minimax/AlphaBeta`):* Hiển thị giá trị lượng giá Minimax tại từng nước đi gốc (`+1.0`, `0.0`, `-1.0` hoặc điểm heuristic).
     - *Cấp 4 (`MCTS`):* Hiển thị tỷ lệ lượt thăm `N(s,a) / N(s)` (`%`).
     - *Cấp 5 (`PolicyNet`):* Hiển thị xác suất Softmax sau khi đã mask ô bất hợp lệ (`%`).
     - *Cấp 6 (`TabularQ / DQN`):* Hiển thị Q-values trực tiếp (hoặc softmax Q-values trên heatmap kèm giá trị Q thô ở bảng Top-5).
     - *Cấp 7 (`HybridMCTS`):* Hiển thị song song **Prior `P(s,a)` từ mạng Policy** và **Visit Share `N(s,a)/N(s)` sau tìm kiếm PUCT** trong bảng Top-5 — giúp minh họa trực quan cách cây tìm kiếm "sửa sai" hoặc "củng cố" gợi ý ban đầu của mạng nơ-ron (**trả lời trực quan cho RQ3**).

### 2.3. Xử lý lỗi nhập liệu & Lệnh tương tác
- Chấp nhận cả hai định dạng nhập thuận tiện: chỉ số ô (ví dụ `5`) hoặc tọa độ hàng-cột (`1,1` hoặc `1 1`).
- Nếu người chơi nhập ô đã đánh, ngoài phạm vi hoặc ký tự rác $\rightarrow$ hiển thị thông báo cảnh báo ngắn gọn màu vàng ngay trên Footer Prompt và giữ nguyên bàn cờ (tuyệt đối không crash).
- Trong chế độ `Human vs AI`, lệnh `u` thực hiện `undo()` 2 bước (hoàn tác nước của AI và nước trước đó của người chơi) để người dùng thử lại thế cờ khi demo.

---

## 3. Thiết kế Hệ thống Biểu đồ Báo cáo (`src/ttt_ai/benchmark/visualizer.py`)

Toàn bộ biểu đồ được xuất tự động từ file log thô (`logs/*.jsonl`) sang thư mục `reports/figures/` ở định dạng `.png` (`dpi=300`, font chữ rõ ràng, không đè nhãn) kèm bảng tổng hợp `.csv` và `.md`.

| Mã Hình | Tên Biểu đồ | Loại Biểu đồ & Bố cục | Phục vụ Mục / Câu hỏi |
|---|---|---|---|
| **Fig 1** | **Ma trận Đối đầu Kép (`X` vs `O`)** | 2 Heatmaps đặt cạnh nhau: Trái = Tỷ lệ thắng & `Score` khi cầm `X`; Phải = khi cầm `O`. Mỗi ô ghi rõ `Score` (hoặc `Win%`) và `(W-D-L)`. | Bảng tổng hợp Round-Robin & **RQ4** |
| **Fig 2** | **Độ trễ Suy luận & Số Nút duyệt** | Biểu đồ cột **thang log** cho `mean_ms` kèm đánh dấu `p95_ms`, và biểu đồ phụ `avg_nodes / avg_simulations` cho 7 cấp AI. | Đánh giá chi phí tính toán (Rubric mục 3.5) |
| **Fig 3** | **Ablation: Policy vs Pure MCTS vs Hybrid MCTS** | Line plot (`Score` kèm vùng tô bóng 95% Bootstrap CI) theo trục `n_simulations ∈ {50, 150, 400}` và theo `time_budget_ms`. Kèm subplot bật/tắt cắt tỉa Alpha-Beta. | Trả lời **RQ3** & Ablation bắt buộc |
| **Fig 4** | **Ablation: Lợi thế Người đi trước ($\Delta\text{Score}$)** | Grouped bar chart kèm error bars 95% CI cho $\Delta\text{Score} = \text{Score}_X - \text{Score}_O$ trên 3 bộ luật: `3×3 K=3`, `4×4 K=3`, và `4×4 K=4`. | Trả lời **RQ4** |
| **Fig 5** | **Learning Curves & Tổng quát hóa Unseen Opponents** | (a) Đường cong học tập (`mean ± std` qua 3 seed) theo epoch/episode; (b) Grouped bar chart so sánh `Supervised Policy` vs `RL Self-play` khi gặp đối thủ unseen (`AlphaBeta`, `MCTS`). | Trả lời **RQ5** |

---

## 4. Quy chuẩn Đóng gói Đầu ra cho Thuyết trình & Báo cáo
- `reports/summary_tables.md` & `reports/summary_metrics.csv`: Tự động xuất đầy đủ `W/D/L`, `Score`, `95% CI`, `mean_ms`, `p95_ms`, `nodes/sims` để sao chép chính xác 100% vào báo cáo LaTeX/Word (đảm bảo quy tắc *Zero Fabrication* trong [RULES.md](./RULES.md)).
- `reports/sample_failure_cases.md`: Trích xuất tự động một số ván thua/hòa đáng chú ý (bàn cờ ASCII + chuỗi nước đi + độ lệch giữa Policy prior và kết quả thực tế) phục vụ phần phân tích lỗi (*Error & Tactical Analysis*, Rubric mục 3.6).
