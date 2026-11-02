import os
import shutil
import subprocess
import sys

DATASET_SLUG = "electraawais/cityscape-dataset"
DEST_DIR = os.path.join(os.path.dirname(__file__), "data", "cityscapes")


def find_dir(root: str, target_name: str) -> str | None:
    for dirpath, dirnames, _ in os.walk(root):
        if target_name in dirnames:
            return os.path.join(dirpath, target_name)
    return None


def link_or_copy(source: str, target: str) -> None:
    if os.path.exists(target):
        print(f"  Đã tồn tại {target}, bỏ qua.")
        return
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", target, source], check=True, shell=False)
    else:
        os.symlink(source, target, target_is_directory=True)


def main() -> None:
    import kagglehub

    os.makedirs(DEST_DIR, exist_ok=True)

    print(f"Đang tải dataset Kaggle '{DATASET_SLUG}' (sẽ hỏi API token nếu chưa cấu hình)...")
    download_root = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Đã tải xong, lưu cache tại: {download_root}")

    img_src = find_dir(download_root, "leftImg8bit")
    gt_src = find_dir(download_root, "gtFine")

    if img_src is None or gt_src is None:
        print("\nKhông tự động tìm thấy thư mục 'leftImg8bit' và/hoặc 'gtFine'.")
        print("Cấu trúc thư mục đã tải về:")
        for dirpath, dirnames, _ in os.walk(download_root):
            depth = dirpath[len(download_root):].count(os.sep)
            if depth <= 2:
                print("  " * depth + os.path.basename(dirpath) + "/")
        print("\nHãy kiểm tra cấu trúc trên và chỉnh lại find_dir()/đường dẫn thủ công.")
        sys.exit(1)

    print(f"Tìm thấy leftImg8bit tại: {img_src}")
    print(f"Tìm thấy gtFine tại     : {gt_src}")

    link_or_copy(img_src, os.path.join(DEST_DIR, "leftImg8bit"))
    link_or_copy(gt_src, os.path.join(DEST_DIR, "gtFine"))

    print(f"\nHoàn tất. Dữ liệu đã sẵn sàng tại: {DEST_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nLỗi: {exc}", file=sys.stderr)
        sys.exit(1)
