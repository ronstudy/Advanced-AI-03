# 04. Nguyên tắc Trung thực Học thuật & Quy ước Kỹ thuật (`RULES.md`)

- **Vai trò tài liệu:** Sổ tay quy ước viết mã, chuẩn kiểm soát thực nghiệm và nguyên tắc trung thực học thuật (Tài liệu số `04` trong lộ trình [PRD.md](./PRD.md) `01 → 06`)
- **Tài liệu liên quan:**
  - Trước: [03. DESIGN.md](./DESIGN.md) — Thiết kế `Rich CLI` và hệ thống biểu đồ
  - Tiếp theo: [05. TASKS.md](./TASKS.md) — Kế hoạch phân rã công việc qua 5 Sprint

---

## 1. Nguyên tắc Trung thực Học thuật (Cốt lõi)

Mục tiêu phần này là giữ cho mọi kết luận trong đồ án đều có bằng chứng thực nghiệm đối chiếu được từ dữ liệu thô trong repository:

1. **Dữ liệu thật từ log thô (*Zero Fabrication*):**
   - Mọi số liệu win-rate, `Score`, độ trễ hay đường cong học tập được báo cáo đều phải phản ánh kết quả chạy thực tế.
   - Các bảng số liệu và biểu đồ trong `reports/` được sinh tự động bằng script phân tích đọc từ `logs/*.jsonl` (hoặc CSV kết xuất từ log), không chỉnh sửa thủ công con số kết quả.
2. **Báo cáo đầy đủ phân phối (*No Cherry-Picking*):**
   - Không chỉ trích riêng các ván thắng đẹp để khái quát hóa kết luận; cần ghi nhận đủ phân phối Thắng/Hòa/Thua trên toàn bộ số ván đã thiết kế và lưu lại cả các ván thua tiêu biểu để phân tích lỗi (*failure mode analysis*).
3. **Chỉ đánh giá mô hình đã huấn luyện:**
   - Các tác tử học máy/học sâu (Cấp 5, 6, 7) khi tham gia giải đấu hoặc trình diễn cần sử dụng trọng số đã train thực tế, đi kèm file `*.meta.json` ghi rõ tên model, kiến trúc, số epoch/episode, `seed`, `val_loss`/`val_acc` (hoặc reward curve), cấu hình bàn cờ và SHA-256 trọng số.
4. **Tránh rò rỉ dữ liệu (*No Data Leakage*):**
   - Với Supervised Learning (Cấp 5), tập dữ liệu được chia `train / validation / test` **theo cấp độ ván đấu (`game_id`)** để các trạng thái trong cùng một ván không xuất hiện đồng thời ở cả tập train lẫn tập test.
   - Không dùng các trận đánh giá cuối cùng (Round-Robin test matches) để tinh chỉnh siêu tham số hoặc chọn điểm dừng (*early stopping*).
5. **Minh bạch nguồn tham khảo & phần tự cài đặt:**
   - Khi kế thừa ý tưởng hoặc đối chiếu cấu trúc từ `TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`, `Tic tac toe Game.pdf`, sách Mark Liu (*AlphaGo Simplified*, 2024) hay các kho mã tham khảo trong `References/`, cần ghi rõ trong docstring và báo cáo phần nào tự cài đặt hoàn toàn và phần nào tham khảo/chuyển thể.

---

## 2. Quy ước Giao diện Game API & Toàn vẹn Trạng thái

1. **Giao tiếp qua 7 phương thức công khai:**
   - Mọi tác tử AI (lớp con của `BaseAgent`) chỉ tương tác với trò chơi thông qua API chuẩn: `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result(player=None)`, `render()`, `clone()` / `undo()`.
   - Tách bạch ranh giới bằng cách không đọc hoặc ghi trực tiếp các thuộc tính nội bộ của engine (`_board`, `_history`, `_winner`) hay trạng thái nội bộ của tác tử đối thủ.
2. **Bảo toàn bàn cờ gốc khi tìm kiếm:**
   - Các tác tử tìm kiếm cây hoặc lookahead (`Minimax`, `AlphaBeta`, `MCTS`, `HybridMCTS`, `RuleBased`) cần thao tác trên bản sao `game.clone()` hoặc thực hiện cặp `step()/undo()` cân bằng để bàn cờ truyền vào `select_action(game)` không bị thay đổi sau khi hàm trả về.
3. **Xử lý nước đi hợp lệ & Action Masking:**
   - Engine kiểm tra hợp lệ trong `step(action)` và ném `IllegalMoveError` nếu `action` ngoài phạm vi `[0, M*N - 1]`, ô đã có quân, hoặc ván đấu đã kết thúc.
   - Các tác tử học sâu / RL (Cấp 5–7) áp dụng **Action Masking** (gán logit `-1e9` / xác suất `0` cho các ô không hợp lệ trước bước `argmax` hoặc lấy mẫu).
4. **Kiểm thử Engine trước khi chạy thực nghiệm:**
   - Trước khi train mô hình hoặc mở giải đấu quy mô lớn, bộ `pytest` cho `engine` cần vượt qua đầy đủ các ca kiểm thử: thắng ngang, dọc, chéo chính, chéo phụ, thắng ở biên/góc, hòa đầy bàn, `IllegalMoveError`, tính độc lập của `clone()` và tính hoàn nguyên của `undo()`.

> **Ghi chú kinh nghiệm (Chống rò rỉ & Kiểm soát trạng thái):**
> - Trong Tic-Tac-Toe, nhiều thế cờ ở giữa ván giống nhau nếu chia ngẫu nhiên theo từng nước đi; tách tập theo `game_id` giúp đánh giá trung thực khả năng tổng quát hóa thay vì mô hình "nhớ vẹt" thế cờ cùng ván.
> - Dùng `tests/test_agents_contract.py` để tự động kiểm tra mọi agent không làm biến đổi `board` gốc và không phát sinh nước đi bất hợp lệ sau khi cài đặt thuật toán mới.

---

## 3. Quy ước Đo lường Công bằng & Tái lập (`Reproducibility`)

1. **Quản lý Seed cố định & Đa Seed:**
   - Các script `train`, `eval`, `tournament` nhận tham số `--seed` và đồng bộ seed cho `random`, `numpy`, và `torch`.
   - Các phương pháp có yếu tố ngẫu nhiên (`Random`, `MCTS`, `Policy` lấy mẫu, `RL`, `HybridMCTS`) được kiểm chứng qua nhiều `random seed` độc lập (ví dụ `42, 123, 999`) để đánh giá độ ổn định.
2. **Công bằng ngân sách tính toán (*Fair Compute Budget*):**
   - Tránh so sánh thuật toán chỉ dựa trên việc cố định cùng độ sâu tìm kiếm; các phép đo đối chứng chính thức báo cáo song song ở **Chế độ Ngân sách Cố định** (cùng `time_budget_ms` hoặc cùng `n_simulations`) và **Chế độ Cấu hình Mặc định**.
   - Trong chế độ giới hạn thời gian (`time_budget_ms`), đồng hồ `time.perf_counter()` đo **toàn bộ thời gian của `select_action()`**, bao gồm tiền xử lý tensor, forward pass của mạng nơ-ron và tìm kiếm cây.
3. **Đo độ trễ sạch & Metadata phần cứng:**
   - Khi đo `mean_ms`, `p95_ms`, `max_ms`, nên hạn chế tác vụ nền nặng gây nhiễu CPU/GPU.
   - Mỗi file log giải đấu tự động lưu dòng metadata đầu tiên gồm: thông số CPU/GPU/RAM, phiên bản Python/PyTorch/NumPy, git commit hash (nếu có) và toàn bộ siêu tham số của từng agent.
4. **Chuẩn thống kê giải đấu:**
   - Mỗi cặp đối đầu Round-Robin thi đấu tối thiểu **100 ván** (chuẩn của dự án là **200 ván/cặp**), cân bằng chính xác **50% đi trước (`X`) và 50% đi sau (`O`)**.
   - Tính `score = (W + 0.5 * D) / N` kèm khoảng tin cậy **95% (Wilson score hoặc Bootstrap)** khi rút ra nhận xét cho RQ1, RQ2, RQ3.

---

## 4. Quy ước Mã nguồn Python (`src/ttt_ai/`)

1. **Phong cách PEP 8 & Định dạng:**
   - Tuân thủ PEP 8 (kiểm tra bằng `ruff`), độ dài dòng tối đa `100` ký tự, đặt tên nhất quán (`snake_case` cho hàm/biến/module, `PascalCase` cho class, `UPPER_SNAKE_CASE` cho hằng số).
2. **Type Hints, Docstrings & Giải thích thuật toán:**
   - Các hàm, phương thức public và lớp cấu hình (`dataclass`) có đầy đủ **Type Annotations** và **Docstring** (Google style: mô tả ngắn, `Args`, `Returns`, `Raises`).
   - Các khối logic toán học quan trọng (`UCT`, `PUCT`, cắt tỉa `Alpha-Beta`, chuẩn hóa `canonical_board`, hàm lượng giá heuristic, cập nhật `Q-value`) có chú thích giải thích lý do (*why*), đặc biệt là quy ước đảo dấu trong trò chơi tổng bằng không.
3. **Tập trung hóa cấu hình:**
   - Gom siêu tham số (`c_uct`, `max_depth`, `learning_rate`, `batch_size`, kích thước `M, N, K`, đường dẫn checkpoint) vào `configs/*.yaml` và `src/ttt_ai/config.py` thay vì viết rải rác trong code thuật toán.
4. **Tách Log khỏi UI & Không lặp logic (*DRY*):**
   - Các gói lõi `engine/`, `agents/`, `training/`, `benchmark/` sử dụng `logging` chuẩn và ghi kết quả ra **JSONL**, tránh gọi `print()` trực tiếp để không làm vỡ bố cục `Rich CLI`.
   - Các hàm lõi như kiểm tra thắng/hòa `M×N, K`, `canonical_board`, `action_masking` đặt tập trung để dùng chung giữa các tác tử, đồng thời giải phóng cây tìm kiếm giữa các ván và giới hạn kích thước replay buffer để giữ ổn định RAM.

> **Ghi chú kinh nghiệm (Ghi log & Tách UI):**
> - Khi `Rich Live` đang chiếm quyền render Terminal ở chế độ `AI vs AI`, bất kỳ lệnh `print()` rải rác nào từ `agents/` hoặc `engine/` đều có thể làm trôi khung dashboard; dùng AST check trong unit test giúp phát hiện sớm lỗi này.

---

## 5. Checklist Tự Kiểm tra Trước Khi Commit hoặc Chạy Thực Nghiệm

- [x] `pytest -v` và `ruff check src/ tests/ scripts/` đã chạy sạch sau lần chỉnh sửa code gần nhất chưa?
- [x] Các bảng số liệu hoặc kết luận mới thêm vào tài liệu có truy xuất trực tiếp được từ `logs/*.jsonl` và `reports/` không?
- [x] Với checkpoint mới train, file `*.meta.json` tương ứng có đầy đủ `board_config`, `seed` và `weights_sha256` chưa?
- [x] Khi đo độ trễ theo `time_budget_ms`, thời gian tiền xử lý tensor và forward pass PyTorch đã được tính vào đồng hồ `perf_counter()` chưa?
- [x] Phần kế thừa từ `References/` và phần tự cài đặt có được chú thích rõ ràng trong docstring và báo cáo không?
