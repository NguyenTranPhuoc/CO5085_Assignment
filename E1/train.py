import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

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
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in history.items():
        axes[0].plot(h["train_loss"], linestyle="--", alpha=0.6, label=f"{name.upper()} Train")
        axes[0].plot(h["val_loss"], linewidth=2, label=f"{name.upper()} Val")
        axes[1].plot([a * 100 for a in h["val_acc"]], linewidth=2, label=f"{name.upper()} Val Acc")

    axes[0].set_title("Learning Curves - Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Validation Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrices(test_predictions, save_path):
    n_models = len(test_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (m_name, (preds, labels)) in zip(axes, test_predictions.items()):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"Confusion Matrix: {m_name.upper()}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_misclassified(model, test_loader, device, save_path, num_samples=10):
    model.eval()
    mis_images, mis_preds, mis_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = preds != labels
            if mask.any():
                for img, p, l in zip(images[mask], preds[mask], labels[mask]):
                    mis_images.append(img.cpu())
                    mis_preds.append(p.item())
                    mis_labels.append(l.item())
                    if len(mis_images) >= num_samples:
                        break
            if len(mis_images) >= num_samples:
                break

    if not mis_images:
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(mis_images):
            ax.axis("off")
            continue
        img_np = mis_images[idx].squeeze(0).numpy() * 0.5 + 0.5
        ax.imshow(img_np, cmap="gray")
        ax.set_title(f"Pred: {CLASS_NAMES[mis_preds[idx]]}\nTrue: {CLASS_NAMES[mis_labels[idx]]}", fontsize=9, color="red")
        ax.axis("off")

    plt.suptitle("Misclassified Samples (Fashion-MNIST)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["softmax", "mlp", "cnn", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs_e1")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: Fashion-MNIST")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, seed=args.seed)

    models_to_run = ["softmax", "mlp", "cnn"] if args.model == "all" else [args.model]
    history, test_predictions, summary_results = {}, {}, []

    for m_name in models_to_run:
        print(f"\n{'='*50}\nTraining: {m_name.upper()}\n{'='*50}")
        model = get_model(m_name, num_classes=len(CLASS_NAMES)).to(device)
        num_params = count_parameters(model)
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
        history[m_name] = h

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        test_predictions[m_name] = (preds, labels)
        print(f"--> Test ({m_name.upper()}): Loss = {test_loss:.4f} | Acc = {test_acc*100:.2f}% | Time = {train_time:.1f}s")
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

        plot_misclassified(model, test_loader, device, os.path.join(args.output_dir, f"misclassified_{m_name}.png"))

        summary_results.append({
            "Model": m_name.upper(),
            "Params": num_params,
            "Best Val Acc (%)": best_val_acc * 100,
            "Test Acc (%)": test_acc * 100,
            "Test Loss": test_loss,
            "Time (s)": train_time
        })

    plot_curves(history, os.path.join(args.output_dir, "learning_curves_fashion_mnist.png"))
    plot_confusion_matrices(test_predictions, os.path.join(args.output_dir, "confusion_matrices_fashion_mnist.png"))

    print(f"\n{'='*65}\nSUMMARY RESULTS (E1 - FASHION-MNIST)\n{'='*65}")
    print(f"{'Model':<10} | {'Params':>12} | {'Val Acc (%)':>12} | {'Test Acc (%)':>12} | {'Time (s)':>10}")
    print("-" * 65)
    for r in summary_results:
        print(f"{r['Model']:<10} | {r['Params']:>12,} | {r['Best Val Acc (%)']:>12.2f} | {r['Test Acc (%)']:>12.2f} | {r['Time (s)']:>10.1f}")


if __name__ == "__main__":
    main()
