# Bài tập E1: Phân loại ảnh với Softmax, MLP và CNN

## 1. Tổng quan

E1 xây dựng và so sánh 3 kiến trúc phân loại ảnh cơ bản trên tập dữ liệu
Fashion-MNIST (28x28, grayscale, 10 lớp):

- **Softmax Classifier**: Bộ phân loại tuyến tính 1 lớp (`Flatten` → `Linear`).
- **MLP Classifier**: Mạng kết nối đầy đủ nhiều tầng (`Linear` → `BatchNorm1d` → `ReLU` → `Dropout`).
- **CNN Classifier**: Mạng tích chập 2D bảo toàn cấu trúc không gian (`Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`).

Mục tiêu là đánh giá độ chính xác, số lượng tham số, đường cong học tập và phân
tích lỗi (confusion matrix, mẫu dự đoán sai) giữa 3 kiến trúc.

## 2. Cấu trúc thư mục

```text
E1/
├── E1_Softmax_MLP_CNN.ipynb   # Notebook chính: data pipeline, model, train, eval
├── model.py                   # Định nghĩa 3 kiến trúc: Softmax, MLP, CNN
├── train.py                   # Script CLI: huấn luyện + đánh giá + xuất hình ảnh kết quả
├── data/                      # Dữ liệu Fashion-MNIST (tự động tải về)
├── outputs_e1/                # Hình ảnh kết quả xuất ra từ notebook/script
└── README.md
```

## 3. Cài đặt môi trường

Yêu cầu:

- Python 3.12
- Jupyter Notebook hoặc JupyterLab
- Các thư viện trong [`requirements.txt`](../requirements.txt) ở thư mục gốc

Cài đặt:

```bash
pip install -r ../requirements.txt
```

Nếu sử dụng GPU, cài phiên bản PyTorch phù hợp với CUDA theo
[hướng dẫn chính thức của PyTorch](https://pytorch.org/get-started/locally/).

## 4. Hướng dẫn chạy

1. Mở [`E1_Softmax_MLP_CNN.ipynb`](./E1_Softmax_MLP_CNN.ipynb) bằng Jupyter/VS Code.
2. Chạy tuần tự các cell từ trên xuống dưới. Dataset `fashion_mnist` sẽ được
   tự động tải về thư mục `./data`.
3. Kết quả (learning curves, confusion matrix, mẫu dự đoán sai) được lưu vào
   thư mục `./outputs_e1`.

### Chạy bằng script `train.py`

Ngoài notebook, có thể huấn luyện trực tiếp qua command line bằng
[`train.py`](./train.py) (dùng chung kiến trúc định nghĩa trong
[`model.py`](./model.py)):

```bash
cd E1
python train.py --model all --epochs 15 --batch_size 64 --lr 1e-3
```

Tham số dòng lệnh:

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--model` | `all` | `softmax`, `mlp`, `cnn` hoặc `all` (huấn luyện cả 3) |
| `--epochs` | `15` | Số epoch huấn luyện |
| `--batch_size` | `64` | Kích thước batch |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--seed` | `42` | Seed để tái lập kết quả |
| `--output_dir` | `./outputs_e1` | Thư mục lưu hình ảnh kết quả |

Script sẽ tự tải dataset về `./data`, in log huấn luyện/đánh giá ra console,
và lưu `learning_curves_fashion_mnist.png`, `confusion_matrices_fashion_mnist.png`,
`misclassified_<model>.png` vào thư mục `--output_dir`.

## 5. Kết quả (Fashion-MNIST, 15 epochs)

| Mô hình | Số tham số | Val Acc (%) | Test Acc (%) | Test Loss |
| :--- | ---: | ---: | ---: | ---: |
| Softmax | 7,850 | 85.17 | 84.20 | 0.4502 |
| MLP | 235,914 | 89.70 | 89.04 | 0.3044 |
| **CNN** | 421,738 | **92.72** | **91.98** | **0.2507** |

CNN đạt độ chính xác cao nhất nhờ bảo toàn cấu trúc không gian 2D; MLP vượt
trội so với Softmax tuyến tính. Nhầm lẫn chủ yếu xảy ra giữa các lớp quần áo
có hình dạng tương tự (Shirt, Pullover, Coat).

## 6. Tài liệu liên quan

- [Notebook E1](./E1_Softmax_MLP_CNN.ipynb)
- [Trang kết quả E1-E3](../docs/exercises.html)
- [README của toàn bộ assignment](../README.md)
