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
├── 2_CNN_vs_ViT_Training.ipynb   # Notebook huấn luyện và đánh giá
├── EDA_Draft/1_EDA.ipynb         # Notebook EDA bản nháp
├── outputs_a1/                    # CSV, JSON và hình ảnh kết quả
└── README.md
```

Các kết quả trực quan hóa được dùng trên landing page nằm trong thư mục
[`docs/`](../docs/). Báo cáo chi tiết được lưu tại
[`Reports/A1_report.pdf`](../Reports/A1_report.pdf).

## 3. Chuẩn bị dữ liệu

Tải và giải nén tập GTSRB từ [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data).
Notebook sử dụng tập dữ liệu GTSRB, ảnh được resize về `224x224` và chia
validation với tỷ lệ 10%. Dữ liệu được tải vào thư mục `data/` theo hướng dẫn
trong notebook.

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

## 5. Kết quả A1.2

- Cấu hình: 3 epochs, batch size 32, seed 42, chạy trên CUDA.
- So sánh ResNet-18 và ViT-B/32 ở các chế độ scratch, frozen, partial và
  finetune.

| Mô hình | Chế độ | Val Acc (%) | Test Acc (%) | F1-macro |
| :--- | :--- | ---: | ---: | ---: |
| ResNet-18 | Scratch | 95.21 | 91.96 | 0.8968 |
| ResNet-18 | Frozen | 78.73 | 68.43 | 0.5991 |
| ResNet-18 | Partial | 99.49 | 96.21 | 0.9462 |
| **ResNet-18** | **Finetune** | **99.90** | **98.64** | **0.9793** |
| ViT-B/32 | Scratch | 25.89 | 23.36 | 0.0951 |
| ViT-B/32 | Frozen | 92.12 | 78.07 | 0.6980 |
| ViT-B/32 | Partial | 98.85 | 94.93 | 0.9072 |
| ViT-B/32 | Finetune | 99.44 | 97.54 | 0.9582 |

Các file kết quả nằm trong [`outputs_a1/`](./outputs_a1/), gồm bảng tổng kết,
log huấn luyện, cấu hình, learning curves và confusion matrices.

## 6. Tài liệu liên quan

- [Notebook huấn luyện](./2_CNN_vs_ViT_Training.ipynb)
- [Notebook EDA bản nháp](./EDA_Draft/1_EDA.ipynb)
- [Trang giới thiệu A1](../docs/a1.html)
- [Báo cáo A1](../Reports/A1_report.pdf)
- [README của toàn bộ assignment](../README.md)
