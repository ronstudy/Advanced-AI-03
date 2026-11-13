# RULES — Quy tắc Kỹ thuật & Trung thực Học thuật

Tài liệu này quy định các **nguyên tắc bắt buộc** trong suốt quá trình thiết kế, lập trình, huấn luyện mô hình, chạy giải đấu và viết báo cáo cho dự án **Game AI Đa Cấp Độ (Extended Tic-Tac-Toe `M×N, K`)**. Mọi mã nguồn và kết quả thực nghiệm vi phạm các quy tắc dưới đây đều bị coi là không đạt chuẩn nghiệm thu.

---

## 1. Quy tắc Trung thực Học thuật & Dữ liệu Thực nghiệm

1. **Cấm bịa đặt hoặc suy diễn số liệu (Zero Fabrication):**
   - Tuyệt đối không báo cáo kết quả benchmark, win-rate, độ trễ hay đường cong học tập khi chưa chạy thực tế.
   - Mọi bảng số liệu và biểu đồ trong báo cáo/slide phải được sinh trực tiếp bằng script phân tích đọc từ log thô (`logs/*.jsonl` hoặc `results/*.csv`). Không chỉnh sửa tay số liệu trong file kết quả.
2. **Cấm chọn lọc mẫu thiên lệch (*No Cherry-Picking*):**
   - Không chỉ chọn các ván thắng đẹp để minh họa kết luận tổng thể. Phải báo cáo toàn bộ phân phối thắng/hòa/thua trên đủ số ván quy định và lưu trữ cả các ván thua để phân tích lỗi (*failure mode analysis*).
3. **Cấm dùng trọng số chưa huấn luyện:**
   - Tuyệt đối không chạy hoặc nộp mạng nơ-ron khởi tạo ngẫu nhiên (chưa train) làm bằng chứng cho các cấp AI học được (Cấp 5, 6, 7).
   - Mọi file checkpoint (`.pt` / `.pth` / `.pkl`) bắt buộc đi kèm metadata ghi rõ: tên model, kiến trúc, ngày giờ train, số epoch/episode, `seed`, `val_loss`/`val_acc` hoặc reward curve, và cấu hình tập dữ liệu.
4. **Chống rò rỉ dữ liệu (*No Data Leakage*):**
   - Với Supervised Learning (Cấp 5), tập dữ liệu phải được chia `train / validation / test` **theo cấp độ ván đấu (`game_id`)** để các trạng thái trong cùng một ván không rơi vào cả tập train lẫn tập test.
   - Tuyệt đối không sử dụng các ván đấu đánh giá cuối cùng (Round-Robin test matches) để chọn lọc siêu tham số hay dừng sớm (*early stopping*).
5. **Minh bạch nguồn tham khảo & kế thừa:**
   - Khi tham khảo ý tưởng hoặc cấu trúc từ `TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`, `Tic tac toe Game.pdf`, sách Mark Liu (*AlphaGo Simplified*, 2024) hay các kho mã nguồn tham khảo trong workspace, phải ghi rõ trong docstring và báo cáo **phần nào do nhóm tự cài đặt hoàn toàn** và **phần nào kế thừa/chuyển thể**.

---

## 2. Quy tắc Hợp đồng API Game & Chống Gian lận

1. **Giao tiếp duy nhất qua Game API chuẩn:**
   - Mọi tác tử AI (`BaseAgent` subclasses) chỉ được phép đọc trạng thái và tương tác với trò chơi thông qua 7 phương thức công khai:
     - `reset()`
     - `legal_actions()`
     - `step(action)`
     - `is_terminal()`
     - `result(player=None)`
     - `render()`
     - `clone()` / `undo()`
   - Nghiêm cấm tác tử truy cập trực tiếp hoặc ghi đè các thuộc tính nội bộ private của engine (như `_board`, `_history`, `_winner`) hoặc trạng thái nội bộ của tác tử đối thủ.
2. **Bảo toàn tính toàn vẹn bàn cờ khi tìm kiếm:**
   - Các tác tử có tìm kiếm cây (`Minimax`, `AlphaBeta`, `MCTS`, `HybridMCTS`, `RuleBased` lookahead) bắt buộc phải thao tác trên bản sao `game.clone()` hoặc thực hiện cặp `step()/undo()` cân bằng hoàn hảo. Bàn cờ gốc truyền vào `select_action(game)` không bao giờ được biến đổi sau khi hàm kết thúc.
3. **Cấm nước đi bất hợp lệ & Hardcode gian lận:**
   - Engine phải kiểm tra tính hợp lệ của mọi nước đi trong `step(action)` và ném ngoại lệ `IllegalMoveError` nếu `action` ngoài phạm vi `[0, M*N - 1]`, ô đã có quân, hoặc ván đấu đã kết thúc.
   - Các tác tử học sâu / RL (Cấp 5–7) bắt buộc phải áp dụng **Action Masking** (gán xác suất `0` / logit `-inf` cho các ô bất hợp lệ trước khi chọn `argmax` hoặc lấy mẫu).
   - Nghiêm cấm hardcode nước đi theo từng thế cờ thử nghiệm (*hardcoded lookup overrides* cho test cases) hoặc lợi dụng logic đặc thù của đối thủ.
4. **Cổng kiểm thử bắt buộc (*Unit Test Gate*):**
   - Trước khi chạy bất kỳ giải đấu hay quy trình train nào, bộ `pytest` cho `engine` phải **pass 100%** các ca kiểm thử: thắng hàng ngang, cột dọc, đường chéo chính, đường chéo phụ, thắng ở ô biên/góc, hòa đầy bàn, ngoại lệ `IllegalMoveError`, tính độc lập bộ nhớ của `clone()`, và tính hoàn nguyên trạng thái của `undo()`.

---

## 3. Quy tắc Đo lường Công bằng & Tái lập (Reproducibility)

1. **Quản lý Seed cố định & Đa Seed:**
   - Mọi script `train`, `eval`, `tournament` phải nhận tham số `--seed` và khóa đồng bộ seed cho `random`, `numpy`, và `torch` (bao gồm deterministic settings nếu áp dụng).
   - Các phương pháp có yếu tố ngẫu nhiên (`Random`, `MCTS`, `Policy` lấy mẫu, `RL`, `HybridMCTS`) khi chạy đánh giá phải kiểm chứng qua nhiều `random seed` độc lập.
2. **Công bằng ngân sách tính toán (*Fair Compute Budget*):**
   - Không kết luận thuật toán này mạnh hơn thuật toán kia chỉ dựa trên "cùng độ sâu tìm kiếm".
   - Mọi phép đo đối chứng chính thức phải báo cáo ở cả **Chế độ Ngân sách Cố định** (cùng `time_budget_ms` hoặc cùng `n_simulations`) và **Chế độ Cấu hình Mặc định**.
   - Trong chế độ giới hạn thời gian (`time_budget_ms`), đồng hồ đo bằng `time.perf_counter()` phải bao trùm **toàn bộ thời gian thực thi của `select_action()`** (tính cả tiền xử lý tensor, forward pass của mạng nơ-ron và tìm kiếm cây).
3. **Đo lường độ trễ sạch & Metadata phần cứng:**
   - Khi chạy benchmark thời gian (`mean_ms`, `p95_ms`, `max_ms`), không chạy đồng thời các tác vụ nặng nền gây nhiễu CPU/GPU.
   - Mỗi phiên giải đấu phải tự động ghi metadata hệ thống vào đầu file log: CPU model, số nhân, GPU (nếu có), dung lượng RAM, phiên bản Python/PyTorch/NumPy, git commit hash (nếu có) và toàn bộ siêu tham số của từng tác tử.
4. **Tính chuẩn xác thống kê:**
   - Mỗi cặp đối đầu trong giải đấu Round-Robin phải thi đấu tối thiểu **100 ván** (mục tiêu chuẩn **200 ván/cặp**), chia chính xác **50% đi trước (`X`) và 50% đi sau (`O`)**.
   - Bắt buộc tính `score = (W + 0.5 * D) / N` và khoảng tin cậy **95% (Wilson hoặc Bootstrap)** trước khi rút ra kết luận so sánh cho RQ3, RQ4, RQ5.

---

## 4. Quy chuẩn Mã nguồn Python & Kiến trúc Kỹ thuật

1. **Chuẩn phong cách PEP 8 & Formatting:**
   - Tuân thủ PEP 8, độ dài dòng tối đa `100` ký tự, đặt tên rõ nghĩa (`snake_case` cho hàm/biến/module, `PascalCase` cho class, `UPPER_SNAKE_CASE` cho hằng số).
2. **Type Hints & Docstrings bắt buộc:**
   - Toàn bộ hàm, phương thức public và class cấu hình (`dataclass` / `TypedDict`) phải có **Type Annotations** đầy đủ.
   - Mọi module, class, và phương thức public phải có **Docstring** (chuẩn Google style: mô tả ngắn, `Args`, `Returns`, `Raises`).
   - Những đoạn logic toán học/thuật toán không hiển nhiên (công thức `UCT`, `PUCT`, cắt tỉa `Alpha-Beta`, chuẩn hóa `canonical_board`, hàm lượng giá heuristic, cập nhật `Q-value`) phải có comment giải thích **tại sao** (*why*), không chỉ mô tả lại dòng code (*what*).
3. **Tách biệt Cấu hình & Logic (*Zero Magic Numbers*):**
   - Không hardcode siêu tham số (`c_uct`, `max_depth`, `learning_rate`, `batch_size`, kích thước bàn `M, N, K`, đường dẫn checkpoint) rải rác trong code logic. Mọi cấu hình phải tập trung tại module/file cấu hình (`configs/*.yaml` hoặc `dataclasses`).
4. **Cơ chế Ghi Log Độc lập & Tách UI:**
   - Lõi `engine/`, `agents/`, `training/`, `benchmark/` **tuyệt đối không dùng `print()` trực tiếp** để tránh xung đột hiển thị với `Rich CLI`.
   - Sử dụng `logging` chuẩn của Python cho thông báo hệ thống và ghi dữ liệu thực nghiệm ra định dạng **JSONL** (mỗi ván đấu / mỗi bước benchmark là một bản ghi độc lập, dễ khôi phục nếu quá trình chạy dài bị gián đoạn).
5. **Không lặp lại logic (*DRY*) & An toàn bộ nhớ:**
   - Hàm kiểm tra thắng/hòa `M×N, K`, hàm chuẩn hóa bàn cờ `canonical_board`, và hàm `action_masking` phải được viết dùng chung một chỗ duy nhất, không copy-paste giữa các agent.
   - Đảm bảo kiểm soát bộ nhớ (giải phóng cây MCTS giữa các ván hoặc giới hạn kích thước replay buffer) để không tràn RAM trong giải đấu hàng nghìn ván.
