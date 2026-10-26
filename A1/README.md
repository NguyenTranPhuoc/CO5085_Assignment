# Bài tập lớn A1: CNN và Vision Transformer

## 1. Tổng quan

A1 thực hiện so sánh hai nhóm kiến trúc thường dùng trong thị giác máy tính:

- **Mạng nơ-ron tích chập (CNN)**.
- **Vision Transformer (ViT)**.

Bài toán được sử dụng là phân loại biển báo giao thông trên tập dữ liệu
[GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data).
Tập dữ liệu có khoảng 51.800 ảnh thuộc 43 lớp, với ảnh có kích thước không đồng nhất
và phân bố mẫu giữa các lớp không cân bằng.

## 2. Cấu trúc thư mục

```text
A1/
├── 1_Notebook/
│   └── 1_EDA.ipynb       # Notebook phân tích và trực quan hóa dữ liệu
└── README.md             
```

Các kết quả trực quan hóa được dùng trên landing page nằm trong thư mục
[`docs/`](../docs/). Báo cáo chi tiết được lưu tại
[`Reports/A1_report.pdf`](../Reports/A1_report.pdf).

## 3. Chuẩn bị dữ liệu

Tải và giải nén tập GTSRB từ [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data).
Notebook hiện tại mong đợi dữ liệu huấn luyện ở vị trí tương đối sau:

## 4. Cài đặt môi trường

Yêu cầu:

- Python 3.12
- Jupyter Notebook hoặc JupyterLab
- PyTorch
- Torchvision
- Matplotlib

Cài đặt các thư viện:

```bash
pip install torch torchvision matplotlib jupyter
```

Nếu sử dụng GPU, hãy cài đặt phiên bản PyTorch phù hợp với CUDA theo
[hướng dẫn chính thức của PyTorch](https://pytorch.org/get-started/locally/).

## 5. Tài liệu liên quan

- [Notebook EDA](./1_Notebook/1_EDA.ipynb)
- [Trang giới thiệu A1](../docs/a1.html)
- [Báo cáo A1](../Reports/A1_report.pdf)
- [README của toàn bộ assignment](../README.md)
