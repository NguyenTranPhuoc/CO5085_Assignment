import torch
import torch.nn as nn
from msa import TransformerEncoderBlock


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size: int = 4, d_model: int = 64):
        super().__init__()
        self.num_tokens = (28 // patch_size) ** 2
        self.proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class PixelTokenizer(nn.Module):
    def __init__(self, downsample: int = 2, d_model: int = 64):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample) if downsample > 1 else nn.Identity()
        eff_size = 28 // downsample
        self.num_tokens = eff_size * eff_size
        self.proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(2).transpose(1, 2)
        return self.proj(x)


class CNNStemTokenizer(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.num_tokens = 7 * 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x).flatten(2).transpose(1, 2)


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer_type: str = "patch",
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        tok = tokenizer_type.lower()
        if tok == "patch":
            self.tokenizer = PatchTokenizer(patch_size=4, d_model=d_model)
        elif tok == "pixel":
            self.tokenizer = PixelTokenizer(downsample=2, d_model=d_model)
        elif tok == "cnn_stem":
            self.tokenizer = CNNStemTokenizer(d_model=d_model)
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer_type}")

        self.num_tokens = self.tokenizer.num_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B = x.size(0)
        tokens = self.tokenizer(x)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        x_seq = self.pos_drop(tokens + self.pos_embed)

        attn_maps = []
        for blk in self.blocks:
            if return_attn:
                x_seq, attn = blk(x_seq, return_attn=True)
                attn_maps.append(attn)
            else:
                x_seq = blk(x_seq)

        x_seq = self.norm(x_seq)
        logits = self.head(x_seq[:, 0])
        if return_attn:
            return logits, attn_maps
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
