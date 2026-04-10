"""
stage2_preprocess/dataset.py
=============================
PyTorch Dataset для обучения LoRA.

Каждый элемент датасета:
    {
        "pixel_values": tensor (3, 512, 512) ∈ [-1, 1],
        "input_ids":    tensor (77,)           — токены CLIPTokenizer,
        "prompt":       str                    — текстовый промпт
    }

Использование:
    from stage2_preprocess.dataset import CelebADataset, build_dataloaders

    train_loader, val_loader = build_dataloaders(
        train_manifest="data/processed/train_proc.json",
        val_manifest="data/processed/val_proc.json",
        batch_size=4,
    )

    for batch in train_loader:
        images     = batch["pixel_values"]   # (B, 3, 512, 512) ∈ [-1, 1]
        input_ids  = batch["input_ids"]      # (B, 77)
        prompts    = batch["prompt"]         # list[str]
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer


# ─── Трансформации ────────────────────────────────────────────────────────────

def make_train_transforms(size: int = 512) -> transforms.Compose:
    """
    Трансформации для обучающей выборки.

    Горизонтальный флип — единственная аугментация.
    ColorJitter, RandomCrop и т.д. НЕ применяем:
      - ColorJitter меняет цвет кожи → противоречит атрибуту
      - RandomCrop может обрезать лицо
    """
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])


def make_val_transforms(size: int = 512) -> transforms.Compose:
    """Трансформации для val/test — без аугментации."""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CelebADataset(Dataset):
    """
    Dataset для LoRA-дообучения Stable Diffusion на CelebA.

    Args:
        manifest_path: путь к JSON-манифесту (train_proc.json / val_proc.json)
        tokenizer_name: имя CLIP-токенизатора
        image_size: размер изображений (512)
        augment: применять горизонтальный флип (только для train)
        max_samples: ограничить число образцов (None = все)

    Формат манифеста (каждая запись):
        {
            "filename":   "000001.jpg",
            "image_path": "data/processed/images/train/000001.png",
            "attrs":      {"Male": 0, "Young": 1, ...},
            "prompt":     "A young woman with blonde hair, smiling, ..."
        }
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        image_size: int = 512,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.image_size = image_size

        # Загружаем манифест
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        if max_samples:
            self.records = self.records[:max_samples]

        # Трансформации
        self.transform = (
            make_train_transforms(image_size)
            if augment
            else make_val_transforms(image_size)
        )

        # CLIP-токенизатор
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """
        Токенизирует текстовый промпт.

        CLIP-токенизатор:
            - max_length=77 (ограничение CLIP)
            - padding="max_length" — дополняем нулями до 77
            - truncation=True — обрезаем если длиннее 77

        Returns:
            input_ids: (77,) long tensor
        """
        tokens = self.tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)  # (77,)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Изображение
        img  = Image.open(rec["image_path"]).convert("RGB")
        pxv  = self.transform(img)   # (3, H, W) ∈ [-1, 1]

        # Токены промпта
        ids  = self._tokenize(rec["prompt"])  # (77,)

        return {
            "pixel_values": pxv,
            "input_ids":    ids,
            "prompt":       rec["prompt"],
        }


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_manifest: Union[str, Path],
    val_manifest:   Union[str, Path],
    batch_size:     int = 4,
    num_workers:    int = 4,
    image_size:     int = 512,
    tokenizer_name: str = "openai/clip-vit-base-patch32",
    max_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт DataLoader'ы для train и val.

    Args:
        batch_size: размер батча НА ОДИН GPU
        num_workers: число параллельных воркеров загрузки
        max_train_samples: None = все данные

    Returns:
        train_loader, val_loader

    Пример:
        train_loader, val_loader = build_dataloaders(
            "data/processed/train_proc.json",
            "data/processed/val_proc.json",
            batch_size=4,
        )
        batch = next(iter(train_loader))
        print(batch["pixel_values"].shape)  # (4, 3, 512, 512)
        print(batch["input_ids"].shape)     # (4, 77)
    """
    train_ds = CelebADataset(
        manifest_path=train_manifest,
        tokenizer_name=tokenizer_name,
        image_size=image_size,
        augment=True,
        max_samples=max_train_samples,
    )
    val_ds = CelebADataset(
        manifest_path=val_manifest,
        tokenizer_name=tokenizer_name,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader


# ─── Быстрая проверка ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    manifest = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train_proc.json"

    print(f"Проверка датасета: {manifest}")
    ds = CelebADataset(manifest, max_samples=10)
    print(f"Размер: {len(ds)}")

    sample = ds[0]
    print(f"pixel_values: {sample['pixel_values'].shape} "
          f"∈ [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
    print(f"input_ids:    {sample['input_ids'].shape}")
    print(f"prompt:       {sample['prompt'][:80]}...")
    print("✓ Dataset работает корректно")
