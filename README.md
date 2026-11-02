# Học sâu và Ứng dụng trong Thị giác Máy tính - CO5085

## Thông tin sinh viên
- Họ và tên: Nguyễn Trần Phước
- MSSV: 2470576
- Email: ntphuoc.sdh242@hcmut.edu.vn

## Giảng viên hướng dẫn
- thầy Lê Thành Sách
- thầy Nguyễn Quốc Minh

## Landing Page:

- Landing Page: https://nguyentranphuoc.github.io/CO5085_Assignment/

## Cấu trúc Repository

```bash
CO5085_Assignment/
│
├── E1/                         # Mã nguồn cho Bài tập E1 
├── E2/                         # Mã nguồn cho Bài tập E2
├── E3/                         # Mã nguồn cho Bài tập E3 
├── A1/                         # Mã nguồn cho Bài tập lớn A1 
├── A2/                         # Mã nguồn cho Bài tập lớn A2 
├── reports/                    # Nơi chứa các file báo cáo PDF 
├── docs/                       # Thư mục mã nguồn cho GitHub Pages
├── README.md                   # Mô tả chung về repo 
├── requirements.txt            # Các thư viện cần cài
```
## Các mốc thời gian nộp bài

| Hạn chót | Tuần | Nội dung công việc | Tiến độ |
| :--- | :--- | :--- | :--- |
| **19/10/2026** | Tuần 7 | Hoàn thành thiết lập khung trang GitHub Pages (Landing page) | **Đã hoàn thành** |
| **26/10/2026** | Tuần 8 | Nộp mốc A1.1 | **Đã hoàn thành** |
| **02/11/2026** | Tuần 9 | Nộp bài tập E1, bài tập E2 và mốc A2.1 | Đang thực hiện |
| **09/11/2026** | Tuần 10 | Nộp bài tập E3 và mốc A1.2| Đang thực hiện |
| **16/11/2026** | Tuần 11 | Nộp mốc A2.2  | Đang thực hiện |

## Hướng dẫn cài đặt môi trường

Yêu cầu:

- Python 3.12
- Jupyter Notebook hoặc JupyterLab (hoặc VS Code với Jupyter extension)

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nếu sử dụng GPU, hãy cài đặt phiên bản `torch` / `torchvision` / `torchaudio`
phù hợp với CUDA theo [hướng dẫn chính thức của PyTorch](https://pytorch.org/get-started/locally/)
thay vì bản CPU mặc định trong `requirements.txt`.

## Hướng dẫn chạy mã nguồn

Mỗi bài tập (E1, E2, E3, A1, A2) có notebook và README riêng mô tả chi tiết
cách chạy và kết quả:

- [E1 - Softmax / MLP / CNN](./E1/README.md)
- [E2 - Transformer, MSA & Tokenization](./E2/README.md)
- [E3 - RNN (LSTM/GRU)](./E3/README.md)
- [A1 - CNN và Vision Transformer](./A1/README.md)
- [A2 - Chủ đề tự chọn + Phân tích Paper](./A2/README.md)

Về tổng thể, các bước chạy chung cho mỗi bài tập là:

1. Cài đặt môi trường theo hướng dẫn ở trên.
2. Mở notebook tương ứng (`.ipynb`) bằng Jupyter hoặc VS Code.
3. Chạy tuần tự các cell từ trên xuống dưới (dataset sẽ tự động được tải về
   thư mục `data/` nếu cần).
4. Kết quả (hình ảnh, bảng số liệu) được lưu vào thư mục `outputs_*` tương
   ứng và có thể xem tổng hợp tại [Landing Page](https://nguyentranphuoc.github.io/CO5085_Assignment/).
