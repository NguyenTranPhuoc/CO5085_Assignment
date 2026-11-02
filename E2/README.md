# Bài tập E2: Transformer — Multi-Head Self-Attention (MSA) và Tokenization

## 1. Tổng quan

E2 hiện thực cơ chế Multi-Head Self-Attention (MSA) từ đầu và thử nghiệm 3
chiến lược tokenization ảnh khác nhau, kết hợp vào một mô hình Vision
Transformer (ViT) đơn giản để phân loại ảnh trên tập Fashion-MNIST:

- **Patch Tokenization**: chia ảnh thành các patch không chồng lấp 4x4
  (n = 49 tokens).
- **Pixel Tokenization**: coi mỗi điểm ảnh (sau gom nhóm 2x2) là một token
  độc lập (n = 196 tokens), minh họa chi phí bậc hai $\mathcal{O}(n^2 d)$.
- **CNN-Stem Tokenization**: dùng mạng tích chập 2 tầng trích xuất đặc trưng
  cục bộ trước khi chuyển thành chuỗi token (n = 49 tokens).

## 2. Cấu trúc thư mục

```text
E2/
├── E2_Transformer_MSA_Tokenization.ipynb   # Notebook chính
├── msa.py                                  # Multi-Head Self-Attention & Transformer Block
├── model.py                                # 3 Tokenizer + Vision Transformer Classifier
├── train.py                                # Script CLI: huấn luyện + đánh giá + xuất hình ảnh kết quả
├── data/                                   # Dữ liệu Fashion-MNIST (tự động tải về)
├── outputs_e2/                             # Hình ảnh kết quả xuất ra từ notebook/script
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

1. Mở [`E2_Transformer_MSA_Tokenization.ipynb`](./E2_Transformer_MSA_Tokenization.ipynb)
   bằng Jupyter/VS Code.
2. Chạy tuần tự các cell từ trên xuống dưới. Dataset `fashion_mnist` sẽ được
   tự động tải về thư mục `./data`.
3. Kết quả (learning curves, attention maps) được lưu vào thư mục `./outputs_e2`.

### Chạy bằng script `train.py`

Ngoài notebook, có thể huấn luyện trực tiếp qua command line bằng
[`train.py`](./train.py) (dùng chung kiến trúc MSA trong [`msa.py`](./msa.py)
và tokenizer/ViT trong [`model.py`](./model.py)):

```bash
cd E2
python train.py --tokenizer all --epochs 15 --d_model 64 --num_heads 4 --num_layers 2
```

Tham số dòng lệnh:

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--tokenizer` | `all` | `patch`, `pixel`, `cnn_stem` hoặc `all` (huấn luyện cả 3) |
| `--d_model` | `64` | Số chiều embedding của Transformer |
| `--num_heads` | `4` | Số head trong Multi-Head Self-Attention |
| `--num_layers` | `2` | Số Transformer Encoder Block |
| `--epochs` | `15` | Số epoch huấn luyện |
| `--batch_size` | `64` | Kích thước batch |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--seed` | `42` | Seed để tái lập kết quả |
| `--output_dir` | `./outputs_e2` | Thư mục lưu hình ảnh kết quả |

Script sẽ tự tải dataset về `./data`, in log huấn luyện/đánh giá ra console,
và lưu `e2_learning_curves_fashion_mnist.png` vào thư mục `--output_dir`.

## 5. Kết quả (Fashion-MNIST, 15 epochs, d_model=64, 2 lớp, 4 heads)

| Tokenizer | Độ dài chuỗi (n) | Số tham số | Val Acc (%) | Test Acc (%) | Test Loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Patch | 50 | 105,098 | 88.55 | 88.30 | 0.3278 |
| Pixel | 197 | 113,546 | 86.20 | 84.72 | 0.3998 |
| **CNN-stem** | 50 | 122,922 | **90.87** | **90.33** | **0.2828** |

CNN-stem cho kết quả tốt nhất nhờ trích xuất đặc trưng cục bộ trước khi đưa
vào MSA. Pixel kém nhất dù có chuỗi dài nhất (n=197), cho thấy chi phí bậc
hai không mang lại lợi thế về độ chính xác. Bản đồ attention của token
`[CLS]` cho thấy mô hình tập trung rõ vào vùng chứa vật thể thay vì nền,
đặc biệt rõ nét hơn ở CNN-stem so với Patch.

## 6. Tài liệu liên quan

- [Notebook E2](./E2_Transformer_MSA_Tokenization.ipynb)
- [Trang kết quả E1-E3](../docs/exercises.html)
- [README của toàn bộ assignment](../README.md)
