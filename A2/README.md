# Bài tập lớn A2: Phân đoạn ngữ nghĩa ảnh đường phố (Cityscapes)

## 1. Chủ đề đã chọn

**Phân đoạn ngữ nghĩa ảnh đường phố (Semantic Segmentation of Urban Scenes)** trên bộ dữ
liệu [Cityscapes](https://www.cityscapes-dataset.com/) — không trùng với bài toán phân loại
ảnh thuần đã thực hiện ở A1.

## 2. Paper phân tích

> Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).
> **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.**
> NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

## 3. Tập dữ liệu & Metric

- **Dataset:** Cityscapes — 2.975 ảnh train / 500 ảnh val / 1.525 ảnh test, độ phân giải
  gốc $2048\times1024$, nhãn `gtFine` quy về 19 lớp "trainId" (+ ignore = 255).
- **Metric đánh giá:** mIoU, Dice, (tùy chọn) FPS.

## 4. Kế hoạch Pipeline (A2.2)

So sánh ≥ 2 phương pháp trên cùng tập Cityscapes:

| Phương pháp | Loại | Backbone |
| :--- | :--- | :--- |
| DeepLabV3+ | CNN | ResNet-50 |
| SegFormer | Transformer | MiT-B0 |

## 5. Cấu trúc thư mục

```text
A2/
├── A2_1_EDA_Dataloader.ipynb   # Mốc A2.1: EDA + Dataset/DataLoader + Augmentation
├── A2_2_Segmentation_Training.ipynb # Mốc A2.2: huấn luyện và đánh giá
├── cityscapes_dataset.py       # Dataset và ánh xạ nhãn Cityscapes
├── download_cityscapes_kaggle.py # Tải dữ liệu từ Kaggle
├── outputs_a2_1/                # Kết quả EDA/DataLoader
├── outputs_a2_2/                # Kết quả huấn luyện/phân đoạn
└── README.md
```

## 6. Chuẩn bị dữ liệu

### Cách 1 (khuyến nghị): Mirror trên Kaggle

Dataset [Cityscape Dataset: Urban Street Scene Segmentation](https://www.kaggle.com/datasets/electraawais/cityscape-dataset)
là bản mirror đầy đủ của Cityscapes (cùng 2.975/500/1.525 ảnh train/val/test, cùng cấu trúc
`leftImg8bit/` + `gtFine/`), chỉ cần tài khoản Kaggle (miễn phí) thay vì tài khoản
cityscapes-dataset.com:

1. Cài `kagglehub`: `pip install kagglehub`.
2. Lấy API token tại [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New
   Token**, đặt file `kaggle.json` vào `~/.kaggle/kaggle.json` (hoặc
   `C:\Users\<user>\.kaggle\kaggle.json` trên Windows).
3. Chạy: `cd A2 && python download_cityscapes_kaggle.py` — script tự tải, tự dò thư mục
   `leftImg8bit`/`gtFine` trong cache Kaggle và tạo NTFS junction (Windows) /
   symlink (Linux/Mac) trỏ vào `A2/data/cityscapes/`.

Sau khi tải, kết quả cuối cùng phải có `A2/data/cityscapes/leftImg8bit/` và
`A2/data/cityscapes/gtFine/`.

Nếu chưa có dữ liệu thật, notebook sẽ tự phát hiện và tạo một tập dữ liệu tổng hợp
(synthetic) nhỏ cùng cấu trúc thư mục để kiểm thử toàn bộ pipeline `Dataset`/`DataLoader`.
Khi dữ liệu thật đã có mặt tại đúng đường dẫn, notebook sẽ tự động dùng dữ liệu thật.

## 7. Cài đặt môi trường

Yêu cầu: Python 3.12, Jupyter, và các thư viện trong
[`requirements.txt`](../requirements.txt) ở thư mục gốc (bao gồm `torch`, `torchvision`
với `transforms.v2`/`tv_tensors`).

```bash
pip install -r ../requirements.txt
```

## 8. Hướng dẫn chạy

Mở [`A2_1_EDA_Dataloader.ipynb`](./A2_1_EDA_Dataloader.ipynb) và chạy tuần tự các cell:

1. Khảo sát dữ liệu (EDA): kích thước ảnh, phân bố pixel theo lớp, mẫu ảnh + mask.
2. Xây dựng `CityscapesSegmentation(Dataset)` tự viết, ánh xạ 34 lớp gốc → 19 trainId.
3. Augmentation đồng bộ ảnh/mask bằng `torchvision.transforms.v2`
   (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, chuẩn hóa ImageNet).
4. Lưu tổng kết mốc A2.1 (`a2_1_summary.json`) vào `outputs_a2_1/`.

## 9. Kết quả A2.2

| Mô hình | Loss | mIoU (%) | mDice (%) | Pixel Acc (%) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **DeepLabV3+-ResNet50** | CE + Dice | **62.77** | **75.36** | **92.79** | 85.5 |
| SegFormer-B0 | CE + Dice | 41.63 | 53.40 | 88.19 | 114.3 |
| SegFormer-B0 | CE | 36.92 | 46.75 | 87.68 | **161.2** |

DeepLabV3+-ResNet50 đạt chất lượng phân đoạn tốt nhất, trong khi SegFormer-B0
với CE thuần có tốc độ suy luận cao nhất. Các file kết quả gồm learning curves,
per-class IoU, dự đoán định tính và bảng CSV trong [`outputs_a2_2/`](./outputs_a2_2/).

## 10. Trạng thái

> **A2.1 (02/11/2026):** đã hoàn thành EDA, Dataset/DataLoader và augmentation.
> **A2.2 (16/11/2026):** đã hoàn thành huấn luyện, so sánh DeepLabV3+/SegFormer
> và tổng hợp kết quả mIoU/Dice.

## 11. Tài liệu liên quan

- [Notebook A2.1](./A2_1_EDA_Dataloader.ipynb)
- [Notebook A2.2](./A2_2_Segmentation_Training.ipynb)
- [Trang giới thiệu A2](../docs/a2.html)
- [README của toàn bộ assignment](../README.md)

