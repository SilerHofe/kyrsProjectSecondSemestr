#!/usr/bin/env python3
"""
stage2_preprocess/preprocess.py
=================================
Этап 2 — предобработка изображений.

Что делает:
  1. Читает манифесты из stage1 (train/val/test.json)
  2. Для каждого изображения:
      - Ресайз до 512×512 (bicubic)
      - Центральная обрезка для сохранения пропорций
  3. Сохраняет обработанные изображения в data/processed/images/
  4. Обновляет пути в манифестах → data/processed/train_proc.json и т.д.

Примечание: img_align_celeba уже выровнен — дополнительного детектирования
лиц НЕ требуется. Просто ресайз + нормировка кадрирования.

Если используете CelebA-HQ (1024×1024) — то же самое, только ресайз.

Запуск:
  python stage2_preprocess/preprocess.py \
      --input  data/processed \
      --output data/processed \
      --size   512 \
      --workers 4
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
from tqdm import tqdm
from rich.console import Console

console = Console()


# ─── Обработка одного изображения ────────────────────────────────────────────

def process_image(
    src_path: Path,
    dst_path: Path,
    size: int = 512,
) -> bool:
    """
    Приводит изображение к квадрату size×size.

    Алгоритм:
        1. Открываем и конвертируем в RGB
        2. Центральная обрезка до квадрата (меньший из W, H)
        3. Ресайз до size×size (LANCZOS — лучшее качество)
        4. Сохраняем как PNG (без потерь)

    Почему центральная обрезка, а не просто ресайз?
        Если ресайзить 178×218 до 512×512 без обрезки,
        получится растянутое лицо. Обрезка сохраняет пропорции.

    Returns:
        True если успешно, False при ошибке
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            img = img.convert("RGB")

            # Центральная обрезка до квадрата
            w, h = img.size
            min_side = min(w, h)
            left  = (w - min_side) // 2
            top   = (h - min_side) // 2
            img   = img.crop((left, top, left + min_side, top + min_side))

            # Ресайз с высоким качеством
            img = img.resize((size, size), Image.LANCZOS)

            # Сохраняем
            img.save(dst_path, "PNG", optimize=False)
        return True

    except Exception as e:
        console.print(f"[red]Ошибка {src_path.name}: {e}[/red]")
        return False


# ─── Пакетная обработка ───────────────────────────────────────────────────────

def process_split(
    records: List[Dict],
    output_images_dir: Path,
    size: int = 512,
    num_workers: int = 4,
    split_name: str = "train",
) -> List[Dict]:
    """
    Обрабатывает список записей параллельно.

    Returns:
        Обновлённые записи с новыми путями к изображениям
    """
    def _task(rec):
        src = Path(rec["image_path"])
        dst = output_images_dir / split_name / src.name.replace(".jpg", ".png")
        ok  = process_image(src, dst, size)
        return ok, dst, rec

    updated  = []
    skipped  = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_task, rec): rec for rec in records}
        for fut in tqdm(as_completed(futures), total=len(records),
                        desc=f"Обработка {split_name}"):
            ok, dst, rec = fut.result()
            if ok:
                new_rec = dict(rec)
                new_rec["image_path"] = str(dst)   # обновляем путь
                updated.append(new_rec)
            else:
                skipped += 1

    if skipped:
        console.print(f"[yellow]Пропущено с ошибками: {skipped}[/yellow]")

    return updated


# ─── Проверка датасета ────────────────────────────────────────────────────────

def verify_dataset(manifest_path: Path, n_check: int = 50) -> None:
    """
    Проверяет случайную выборку изображений из манифеста.
    Убеждается, что все файлы существуют и имеют правильный размер.
    """
    with open(manifest_path) as f:
        records = json.load(f)

    import random
    sample = random.sample(records, min(n_check, len(records)))

    errors = 0
    for rec in sample:
        p = Path(rec["image_path"])
        if not p.exists():
            console.print(f"[red]НЕТ ФАЙЛА: {p}[/red]")
            errors += 1
            continue
        try:
            with Image.open(p) as img:
                if img.size != (512, 512):
                    console.print(f"[yellow]Неверный размер {img.size}: {p}[/yellow]")
                    errors += 1
        except Exception as e:
            console.print(f"[red]Ошибка чтения {p}: {e}[/red]")
            errors += 1

    if errors == 0:
        console.print(f"[green]✓ Проверка пройдена ({n_check} изображений)[/green]")
    else:
        console.print(f"[red]✗ Ошибок: {errors} из {n_check}[/red]")


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Предобработка изображений CelebA")
    parser.add_argument("--input",   default="data/processed",
                        help="Папка с манифестами (train.json, val.json, test.json)")
    parser.add_argument("--output",  default="data/processed",
                        help="Папка для обработанных изображений")
    parser.add_argument("--size",    type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits",  nargs="+", default=["train", "val", "test"])
    parser.add_argument("--verify",  action="store_true", default=True)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Предобработка изображений CelebA → {args.size}×{args.size}[/bold cyan]")

    for split in args.splits:
        manifest = input_dir / f"{split}.json"
        if not manifest.exists():
            console.print(f"[yellow]Пропуск: {manifest} не найден[/yellow]")
            continue

        with open(manifest) as f:
            records = json.load(f)

        console.print(f"\n[bold]{split}: {len(records):,} изображений[/bold]")

        updated = process_split(
            records=records,
            output_images_dir=images_dir,
            size=args.size,
            num_workers=args.workers,
            split_name=split,
        )

        # Сохраняем обновлённый манифест
        out_manifest = output_dir / f"{split}_proc.json"
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)

        console.print(f"[green]✓ {split}_proc.json → {len(updated):,} записей[/green]")

        # Проверка
        if args.verify:
            verify_dataset(out_manifest)

    console.print("\n[bold green]Предобработка завершена![/bold green]")
    console.print(f"Изображения: {images_dir}")
    console.print("\nСледующий шаг:")
    console.print("  python stage3_train/train_lora.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
