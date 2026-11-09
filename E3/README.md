# Bài tập E3: LSTM / GRU trên chuỗi ảnh

## 1. Tổng quan

E3 mô hình hóa ảnh Fashion-MNIST (28x28, grayscale, 10 lớp) như **dữ liệu
chuỗi**, hiện thực cả `nn.LSTM` và `nn.GRU`, và so sánh với baseline không
hồi quy (MLP/CNN từ E1) trên cùng tập dữ liệu/split.

Hai cách biểu diễn ảnh thành chuỗi được thử nghiệm và so sánh:

- **Row sequence**: mỗi hàng pixel (28 giá trị) là một bước thời gian →
  chuỗi dài 28, mỗi bước là vector 28 chiều.
- **Patch sequence**: chia ảnh thành các patch 4x4 không chồng lấp, quét
  theo thứ tự hàng/cột (raster scan) → chuỗi dài 49, mỗi bước là vector 16
  chiều.

Với mỗi cách biểu diễn, cả LSTM và GRU đều được huấn luyện và đánh giá, sau
đó so sánh với MLP và CNN (kiến trúc giống hệt E1) về accuracy, số tham số,
và khả năng bắt phụ thuộc thời gian (temporal dependency).

### Thí nghiệm độ nhạy thứ tự chuỗi (temporal dependency)

Sau khi huấn luyện, mỗi mô hình được đánh giá lại trên tập test đã **đảo
ngược thứ tự hàng pixel** (giữ nguyên nội dung từng hàng, chỉ đổi thứ tự
hàng trên-dưới). Vì LSTM/GRU học cách tổng hợp thông tin theo đúng thứ tự
bước thời gian đã thấy lúc huấn luyện, việc đảo thứ tự này phá vỡ giả định
tuần tự và dự kiến làm giảm accuracy mạnh hơn so với MLP/CNN (vốn không mô
hình hóa thứ tự theo chiều thời gian một cách tường minh).

## 2. Cấu trúc thư mục

```text
E3/
├── E3_LSTM_GRU_Sequence.ipynb   # Notebook chính
├── model.py                     # RecurrentClassifier (LSTM/GRU) + baseline MLP/CNN
├── train.py                     # Script CLI: huấn luyện + đánh giá + xuất hình ảnh kết quả
├── data/                        # Dữ liệu Fashion-MNIST (tự động tải về)
├── outputs_e3/                  # Hình ảnh/kết quả xuất ra từ notebook/script
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

1. Mở [`E3_LSTM_GRU_Sequence.ipynb`](./E3_LSTM_GRU_Sequence.ipynb) bằng
   Jupyter/VS Code.
2. Chạy tuần tự các cell từ trên xuống dưới. Dataset `fashion_mnist` sẽ được
   tự động tải về thư mục `./data`.
3. Kết quả (learning curves, bảng so sánh, độ nhạy thứ tự chuỗi) được lưu
   vào thư mục `./outputs_e3`.

### Chạy bằng script `train.py`

Ngoài notebook, có thể huấn luyện trực tiếp qua command line bằng
[`train.py`](./train.py) (dùng chung kiến trúc trong [`model.py`](./model.py)):

```bash
cd E3
python train.py --sequence all --epochs 15 --hidden_size 128 --num_layers 1
```

Tham số dòng lệnh:

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--sequence` | `all` | `row`, `patch` hoặc `all` (chạy cả 2 cách biểu diễn) |
| `--hidden_size` | `128` | Số chiều hidden state của LSTM/GRU |
| `--num_layers` | `1` | Số lớp xếp chồng của LSTM/GRU |
| `--epochs` | `15` | Số epoch huấn luyện |
| `--batch_size` | `64` | Kích thước batch |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--seed` | `42` | Seed để tái lập kết quả |
| `--output_dir` | `./outputs_e3` | Thư mục lưu hình ảnh kết quả |
| `--skip_baseline` | `False` | Bỏ qua huấn luyện lại baseline MLP/CNN |

Script sẽ huấn luyện baseline MLP/CNN, sau đó LSTM/GRU trên các cách biểu
diễn được chọn, in log huấn luyện/đánh giá ra console (bao gồm bảng độ
nhạy thứ tự chuỗi), và lưu `e3_learning_curves_fashion_mnist.png`,
`e3_summary_bars_fashion_mnist.png` vào thư mục `--output_dir`.

## 5. Kết quả (Fashion-MNIST, 15 epochs, hidden_size=128, 1 lớp)

> Điền lại số liệu thực tế sau khi chạy `train.py`/notebook trên máy của bạn.

| Mô hình | Số tham số | Val Acc (%) | Test Acc (%) | Test Loss |
| :--- | ---: | ---: | ---: | ---: |
| MLP | 235,914 | — | — | — |
| CNN | 421,738 | — | — | — |
| LSTM-row | — | — | — | — |
| GRU-row | — | — | — | — |
| LSTM-patch | — | — | — | — |
| GRU-patch | — | — | — | — |

### Độ nhạy thứ tự chuỗi (đảo hàng pixel lúc test)

| Mô hình | Test Acc gốc (%) | Test Acc đảo hàng (%) | Giảm (điểm %) |
| :--- | ---: | ---: | ---: |
| MLP | — | — | — |
| CNN | — | — | — |
| LSTM-row | — | — | — |
| GRU-row | — | — | — |
| LSTM-patch | — | — | — |
| GRU-patch | — | — | — |

GRU và LSTM thường đạt accuracy thấp hơn CNN (do không bảo toàn cấu trúc
không gian 2D như CNN) nhưng có thể cạnh tranh với MLP; biểu diễn patch
(chuỗi ngắn hơn, mỗi bước mang nhiều thông tin không gian cục bộ hơn) nhìn
chung cho kết quả tốt hơn biểu diễn row. Khi đảo thứ tự hàng lúc test,
LSTM/GRU giảm accuracy rõ rệt hơn MLP/CNN, cho thấy các mô hình hồi quy
thực sự khai thác phụ thuộc theo thứ tự thời gian của chuỗi.

## 6. Tài liệu liên quan

- [Notebook E3](./E3_LSTM_GRU_Sequence.ipynb)
- [Trang kết quả E1-E3](../docs/exercises.html)
- [README của toàn bộ assignment](../README.md)
