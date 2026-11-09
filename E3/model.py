import torch
import torch.nn as nn

PATCH_SIZE = 4


def image_to_row_sequence(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, seq_len=28, input_size=28), mỗi hàng ảnh là 1 bước thời gian."""
    return x.squeeze(1)


def image_to_patch_sequence(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, seq_len=49, input_size=16), quét patch 4x4 theo thứ tự hàng/cột."""
    b = x.size(0)
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.contiguous().view(b, -1, patch_size * patch_size)


class RecurrentClassifier(nn.Module):
    """Phân loại ảnh bằng cách coi ảnh như chuỗi (hàng pixel hoặc chuỗi patch), dùng nn.LSTM/nn.GRU."""

    def __init__(
        self,
        rnn_type: str = "lstm",
        sequence_type: str = "row",
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        if sequence_type not in ("row", "patch"):
            raise ValueError(f"Unknown sequence_type: {sequence_type}")
        self.sequence_type = sequence_type
        input_size = 28 if sequence_type == "row" else PATCH_SIZE * PATCH_SIZE

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(rnn_type.lower())
        if rnn_cls is None:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_type == "row":
            return image_to_row_sequence(x)
        return image_to_patch_sequence(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.to_sequence(x)
        out, _ = self.rnn(seq)
        last_step = out[:, -1, :]
        return self.classifier(last_step)


class MLPClassifier(nn.Module):
    """Baseline không hồi quy (tái sử dụng kiến trúc từ E1)."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))


class CNNClassifier(nn.Module):
    """Baseline không hồi quy (tái sử dụng kiến trúc từ E1)."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    name: str,
    sequence_type: str = "row",
    num_classes: int = 10,
    hidden_size: int = 128,
    num_layers: int = 1
) -> nn.Module:
    key = name.lower()
    if key in ("lstm", "gru"):
        return RecurrentClassifier(
            rnn_type=key, sequence_type=sequence_type,
            hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes
        )
    if key == "mlp":
        return MLPClassifier(num_classes)
    if key == "cnn":
        return CNNClassifier(num_classes)
    raise ValueError(f"Unknown model: {name}")
