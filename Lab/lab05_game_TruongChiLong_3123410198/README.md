# Lab 05: Tìm kiếm đối kháng và Trò chơi (Adversarial Search & Games)

> **Môn học:** Trí tuệ nhân tạo nâng cao (Advanced Artificial Intelligence)  
> **Sinh viên thực hiện:** Trương Chí Long  
> **MSSV:** 3123410198  

---

## 1. Yêu cầu môi trường & Cài đặt

Dự án phát triển trên nền tảng Python 3 (khuyên dùng Python 3.10+).

Cài đặt các thư viện phụ thuộc bằng `pip`:

```bash
pip install numpy matplotlib pandas notebook ipykernel
```

---

## 2. Cấu trúc thư mục

```text
lab05_game_TruongChiLong_3123410198/
├── tictactoe.py                                      # Module logic & hàm tiện ích game Tic-Tac-Toe
├── tictactoe_definitions.ipynb                       # Định nghĩa bài toán & không gian trạng thái Tic-Tac-Toe
├── tictactoe_and_or_tree_search.ipynb                # Giải Tic-Tac-Toe bằng AND-OR Tree Search
├── tictactoe_alpha_beta_tree_search.ipynb            # Minimax kết hợp cắt tỉa Alpha-Beta
├── tictactoe_heuristic_alpha_beta_tree_search.ipynb  # Heuristic Alpha-Beta Search (độ sâu giới hạn)
├── tictactoe_pure_monte_carlo_search.ipynb           # Pure Monte Carlo Search (mô phỏng ngẫu nhiên)
├── tictactoe_monte_carlo_tree_search_restricted.ipynb# Monte Carlo Tree Search (MCTS) với chính sách UCB1
├── tictactoe_interactive.ipynb                       # Giao diện chơi tương tác trực tiếp với Agent
├── assignment_connect4.ipynb                         # Bài tập 1: Giải thuật đối kháng chơi Connect 4
├── assignment_mean_connect4.ipynb                    # Bài tập 2: Đối kháng biến thể "Mean" Connect 4
├── assignment_dots_and_boxes.ipynb                   # Bài tập 3: Mô hình hóa & giải thuật game Dots and Boxes
├── profiling_code.ipynb                              # Đo lường thời gian & profiling hiệu năng
└── store_agent_state_information.ipynb               # Cơ chế lưu trữ trạng thái agent qua các lượt
```

---

## 3. Hướng dẫn cách chạy

### Cách 1: Chạy bằng Jupyter Notebook / JupyterLab (Khuyên dùng)

1. Mở terminal và di chuyển vào thư mục dự án:
   ```bash
   cd lab05_game_TruongChiLong_3123410198
   ```
2. Khởi chạy Jupyter:
   ```bash
   jupyter notebook
   ```
   *(hoặc `jupyter lab`)*
3. Mở notebook cần chạy (ví dụ `assignment_connect4.ipynb` hoặc `tictactoe_interactive.ipynb`), sau đó chọn **Kernel** → **Restart & Run All** (hoặc chạy lần lượt từng cell từ trên xuống dưới).

### Cách 2: Chạy trên Visual Studio Code / Cursor

1. Mở thư mục dự án trong VS Code.
2. Cài đặt các extension: **Python** và **Jupyter**.
3. Mở file `.ipynb` bất kỳ, chọn đúng Python Environment / Kernel đã cài đặt thư viện.
4. Bấm **Run All** hoặc thực thi từng cell bằng tổ hợp phím `Shift + Enter`.

### Cách 3: Chạy trên Google Colab

1. Tải các file `.ipynb` lên Google Colab.
2. Tải kèm file `tictactoe.py` lên cùng thư mục làm việc trên Colab (bắt buộc đối với các notebook Tic-Tac-Toe).
3. Chọn **Runtime** → **Run all** để chạy toàn bộ notebook.

---

## 4. Lưu ý khi thực thi

- **Thư mục làm việc (Working Directory):** File `tictactoe.py` phải nằm cùng thư mục làm việc với các notebook để tránh lỗi `ModuleNotFoundError: No module named 'tictactoe'`.
- **Chế độ chơi tương tác (`tictactoe_interactive.ipynb`):** Khi đến lượt người chơi, nhập chỉ số ô hợp lệ (từ `0` đến `8`) vào ô nhập liệu (input box) tương ứng với vị trí hiển thị trên bàn cờ.
- **Thời gian chạy thuật toán:** Các thuật toán duyệt cây sâu hoặc MCTS với số vòng mô phỏng lớn có thể mất vài phút để hoàn thành việc tính toán nước đi tối ưu.
