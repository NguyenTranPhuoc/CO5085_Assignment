import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ================= CẤU HÌNH HẰNG SỐ CITYSCAPES =================
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]
NUM_CLASSES = 19
IGNORE_INDEX = 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

# Bảng tra cứu nhanh 256 phần tử (labelId -> trainId)
LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for k, v in ID_TO_TRAINID.items():
    LABEL_LUT[k] = v


def encode_segmap(mask_np: np.ndarray) -> np.ndarray:
    """Chuyển đổi mask labelId gốc của Cityscapes sang 19 trainId (0..18 và 255)."""
    return LABEL_LUT[mask_np]


def decode_segmap(trainid_mask: np.ndarray) -> np.ndarray:
    """Chuyển đổi mặt nạ trainId (0..18) sang ảnh màu RGB theo bảng màu chuẩn Cityscapes."""
    h, w = trainid_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        rgb[trainid_mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
    return rgb


class CityscapesDataset(Dataset):
    """Dataset phân đoạn ngữ nghĩa Cityscapes (an toàn với num_workers > 0 trên Windows)."""

    def __init__(self, root_dir: str, split: str = "train", crop_size=(256, 512), augment: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        self.samples = []

        img_root = os.path.join(root_dir, "leftImg8bit", split)
        gt_root = os.path.join(root_dir, "gtFine", split)

        if os.path.exists(img_root):
            for city in sorted(os.listdir(img_root)):
                city_img_dir = os.path.join(img_root, city)
                city_gt_dir = os.path.join(gt_root, city)
                if not os.path.isdir(city_img_dir):
                    continue
                for fname in sorted(os.listdir(city_img_dir)):
                    if fname.endswith("_leftImg8bit.png"):
                        prefix = fname.replace("_leftImg8bit.png", "")
                        img_path = os.path.join(city_img_dir, fname)
                        gt_path = os.path.join(city_gt_dir, f"{prefix}_gtFine_labelIds.png")
                        if os.path.exists(gt_path):
                            self.samples.append((img_path, gt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(gt_path)

        if self.augment:
            # Dùng T.RandomResizedCrop.get_params (từ torchvision.transforms)
            i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1.8, 2.2))
            image = TF.resized_crop(image, i, j, h, w, self.crop_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.crop_size, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        else:
            image = TF.resize(image, self.crop_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.crop_size, interpolation=TF.InterpolationMode.NEAREST)

        img_tensor = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_np = encode_segmap(np.array(mask, dtype=np.uint8))
        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, mask_tensor
