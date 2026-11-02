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
├── data/cityscapes/            # Dữ liệu Cityscapes (tự tải thủ công, xem mục 6)
├── outputs_a2_1/                # Hình ảnh/kết quả EDA xuất ra từ notebook
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

### Cách 2: Trang chủ Cityscapes chính thức

1. Đăng ký và đăng nhập tại [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).
2. Chạy `python download_cityscapes.py` (script sẽ hỏi username/password ngay trong
   terminal) để tự tải và giải nén `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`.

Dù dùng cách nào, kết quả cuối cùng phải có `A2/data/cityscapes/leftImg8bit/` và
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

## 9. Trạng thái

> **A2.1 (hạn 02/11/2026):** đã khóa chủ đề, paper, mô tả dataset/metric, và hoàn thành
> EDA + Dataset/DataLoader + augmentation.
> **A2.2 (hạn 16/11/2026):** sẽ triển khai huấn luyện DeepLabV3+ và SegFormer, so sánh số
> liệu (mIoU/Dice) và phân tích paper đầy đủ.

## 10. Tài liệu liên quan

- [Notebook A2.1](./A2_1_EDA_Dataloader.ipynb)
- [Trang giới thiệu A2](../docs/a2.html)
- [README của toàn bộ assignment](../README.md)

