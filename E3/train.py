import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

from model import get_model, count_parameters

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64, val_ratio: float = 0.1, seed: int = 42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size = int(len(full_train_ds) * val_ratio)
    train_size = len(full_train_ds) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, transform_fn=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if transform_fn is not None:
            images = transform_fn(images)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def flip_rows(images: torch.Tensor) -> torch.Tensor:
    """Đảo thứ tự các hàng (chiều cao) của ảnh -> phá vỡ thứ tự chuỗi mà RNN đã học."""
    return torch.flip(images, dims=[2])


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in history.items():
        axes[0].plot(h["train_loss"], linestyle="--", alpha=0.6, label=f"{name.upper()} Train")
        axes[0].plot(h["val_loss"], linewidth=2, label=f"{name.upper()} Val")
        axes[1].plot([a * 100 for a in h["val_acc"]], linewidth=2, label=f"{name.upper()} Val Acc")

    axes[0].set_title("E3 Learning Curves - Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_summary_bars(summary_results, save_path):
    names = [r["Model"] for r in summary_results]
    accs = [r["Test Acc (%)"] for r in summary_results]
    params = [r["Params"] for r in summary_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, accs, color="steelblue")
    axes[0].set_title("Test Accuracy (%)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    axes[1].bar(names, params, color="darkorange")
    axes[1].set_title("Trainable Parameters")
    axes[1].set_ylabel("Params")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="all", choices=["row", "patch", "all"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs_e3")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: Fashion-MNIST")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, seed=args.seed)

    sequence_types = ["row", "patch"] if args.sequence == "all" else [args.sequence]
    recurrent_combos = [(rnn, seq) for seq in sequence_types for rnn in ("lstm", "gru")]
    baseline_names = [] if args.skip_baseline else ["mlp", "cnn"]

    history, summary_results, order_sensitivity = {}, [], []

    def run_model(model_key: str, sequence_type: str, display_name: str):
        model = get_model(
            model_key, sequence_type=sequence_type, num_classes=len(CLASS_NAMES),
            hidden_size=args.hidden_size, num_layers=args.num_layers
        ).to(device)
        num_params = count_parameters(model)
        print(f"\n{'='*55}\nTraining: {display_name}\n{'='*55}")
        print(f"Trainable Parameters: {num_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc, best_state = 0.0, None
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

            h["train_loss"].append(tr_loss)
            h["val_loss"].append(va_loss)
            h["train_acc"].append(tr_acc)
            h["val_acc"].append(va_acc)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:.2f}%")

        train_time = time.time() - start_time
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        print(f"--> Test ({display_name}): Loss = {test_loss:.4f} | Acc = {test_acc*100:.2f}% | Time = {train_time:.1f}s")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

        # Kiểm tra độ nhạy với thứ tự chuỗi: đánh giá lại trên tập test đã đảo hàng
        _, flipped_acc, _, _ = evaluate(model, test_loader, criterion, device, transform_fn=flip_rows)

        history[display_name] = h
        summary_results.append({
            "Model": display_name,
            "Params": num_params,
            "Best Val Acc (%)": round(best_val_acc * 100, 2),
            "Test Acc (%)": round(test_acc * 100, 2),
            "Test Loss": round(test_loss, 4),
            "Time (s)": round(train_time, 1)
        })
        order_sensitivity.append({
            "Model": display_name,
            "Original Test Acc (%)": round(test_acc * 100, 2),
            "Row-Flipped Test Acc (%)": round(flipped_acc * 100, 2),
            "Drop (pp)": round((test_acc - flipped_acc) * 100, 2)
        })

    for b_name in baseline_names:
        run_model(b_name, sequence_type="row", display_name=b_name.upper())

    for rnn_type, seq_type in recurrent_combos:
        run_model(rnn_type, sequence_type=seq_type, display_name=f"{rnn_type.upper()}-{seq_type}")

    plot_curves(history, os.path.join(args.output_dir, "e3_learning_curves_fashion_mnist.png"))
    plot_summary_bars(summary_results, os.path.join(args.output_dir, "e3_summary_bars_fashion_mnist.png"))

    print(f"\n{'='*80}\nSUMMARY RESULTS (E3 - FASHION-MNIST)\n{'='*80}")
    print(f"{'Model':<14} | {'Params':>10} | {'Val Acc (%)':>12} | {'Test Acc (%)':>12} | {'Time (s)':>8}")
    print("-" * 80)
    for r in summary_results:
        print(f"{r['Model']:<14} | {r['Params']:>10,} | {r['Best Val Acc (%)']:>12.2f} | {r['Test Acc (%)']:>12.2f} | {r['Time (s)']:>8.1f}")

    print(f"\n{'='*80}\nORDER SENSITIVITY (đảo hàng pixel lúc test)\n{'='*80}")
    print(f"{'Model':<14} | {'Original (%)':>12} | {'Row-Flipped (%)':>16} | {'Drop (pp)':>10}")
    print("-" * 80)
    for r in order_sensitivity:
        print(f"{r['Model']:<14} | {r['Original Test Acc (%)']:>12.2f} | {r['Row-Flipped Test Acc (%)']:>16.2f} | {r['Drop (pp)']:>10.2f}")


if __name__ == "__main__":
    main()
