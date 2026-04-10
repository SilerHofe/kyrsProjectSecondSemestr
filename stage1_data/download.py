#!/usr/bin/env python3
"""
stage1_data/download.py
========================
Скачивание датасета CelebA.

Google Drive (gdown) часто блокирует скачивание крупных файлов —
ошибка "Too many users have viewed or downloaded this file recently".

Методы (в порядке рекомендации):
  1. kaggle      — надёжно, быстро, нужна бесплатная регистрация
  2. huggingface — надёжно, без регистрации
  3. manual      — вручную через браузер (всегда работает)
  4. test        — тестовые данные (100 изображений) для проверки пайплайна

Запуск:
  python stage1_data/download.py                    # интерактивный выбор
  python stage1_data/download.py --method kaggle
  python stage1_data/download.py --method huggingface
  python stage1_data/download.py --method test      # только для отладки
  python stage1_data/download.py --check            # проверить скачанное
"""

import os
import sys
import random
import shutil
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

# ─── Проверка скачанного ──────────────────────────────────────────────────────

def check_existing(output_dir: Path) -> bool:
    """Проверяет наличие всех нужных файлов, возвращает True если всё есть."""
    checks = {
        "images":    output_dir / "img_align_celeba",
        "attrs":     output_dir / "list_attr_celeba.txt",
        "partition": output_dir / "list_eval_partition.txt",
    }
    all_ok = True
    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                n = len(list(path.glob("*.jpg")))
                console.print(f"  [green]✓[/green] {name}: {path} ({n:,} изображений)")
                if n < 100:
                    all_ok = False
            else:
                size_kb = path.stat().st_size // 1024
                console.print(f"  [green]✓[/green] {name}: {path} ({size_kb:,} KB)")
        else:
            console.print(f"  [red]✗[/red] {name}: НЕ НАЙДЕНО ({path})")
            all_ok = False
    return all_ok


# ─── Метод 1: Kaggle (РЕКОМЕНДУЕТСЯ) ─────────────────────────────────────────

def download_via_kaggle(output_dir: Path) -> bool:
    """
    Скачивает CelebA через Kaggle API.
    Dataset: jessicali9530/celeba-dataset (~1.4 GB)
    Содержит изображения + все аннотации.

    Предварительно:
      1. Зарегистрируйтесь бесплатно на kaggle.com
      2. Профиль → Settings → API → Create New Token → скачается kaggle.json
      3. Windows: поместите в C:\\Users\\<ВашеИмя>\\.kaggle\\kaggle.json
         Linux/Mac: ~/.kaggle/kaggle.json
      4. pip install kaggle
    """
    # Проверяем наличие kaggle
    try:
        import kaggle
    except ImportError:
        console.print("[red]kaggle не установлен.[/red]")
        console.print("Запустите: [bold]pip install kaggle[/bold]")
        return False

    # Проверяем credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        console.print(Panel(
            "[bold]Файл kaggle.json не найден![/bold]\n\n"
            "Как получить:\n"
            "  1. Зарегистрируйтесь на [link=https://kaggle.com]kaggle.com[/link]\n"
            "  2. Профиль → Settings → API → [bold]Create New Token[/bold]\n"
            "  3. Скачается kaggle.json\n"
            f"  4. Поместите в: [bold]{kaggle_json}[/bold]\n"
            "  5. Запустите снова: python stage1_data/download.py --method kaggle",
            title="Настройка Kaggle API",
            border_style="yellow",
        ))
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Скачивание CelebA через Kaggle (~1.4 GB)...[/cyan]")
    console.print("  Dataset: jessicali9530/celeba-dataset")

    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "jessicali9530/celeba-dataset",
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        _fix_kaggle_structure(output_dir)
        console.print("[green]✓ CelebA скачан через Kaggle![/green]")
        return True

    except Exception as e:
        console.print(f"[red]Ошибка Kaggle: {e}[/red]")
        return False


def _fix_kaggle_structure(output_dir: Path):
    """Kaggle иногда создаёт вложенные папки — нормализуем структуру."""
    for sub in ["img_align_celeba"]:
        for found in list(output_dir.rglob(sub)):
            target = output_dir / sub
            if found.is_dir() and found != target:
                if not target.exists():
                    shutil.move(str(found), str(target))
                    console.print(f"  Перемещено: {found.name} → {target}")

    for txt in ["list_attr_celeba.txt", "list_eval_partition.txt",
                "list_bbox_celeba.txt", "list_landmarks_align_celeba.txt"]:
        for found in list(output_dir.rglob(txt)):
            target = output_dir / txt
            if found != target and not target.exists():
                shutil.move(str(found), str(target))


# ─── Метод 2: HuggingFace ─────────────────────────────────────────────────────

def download_via_huggingface(output_dir: Path) -> bool:
    """
    Скачивает CelebA через HuggingFace Hub.

    Изображения: "huggan/CelebA-HQ-256" (256px, 30 000 изображений)
    Аннотации:   скачиваем txt-файлы напрямую
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Шаг 1: Аннотации через HuggingFace Hub ────────────────────────────
    console.print("[cyan]Скачивание аннотаций через HuggingFace Hub...[/cyan]")
    _download_annotations_hf(output_dir)

    # ── Шаг 2: Изображения ────────────────────────────────────────────────
    img_dir = output_dir / "img_align_celeba"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 1000:
        console.print(f"[yellow]Изображения уже скачаны: {img_dir}[/yellow]")
        return True

    console.print("[cyan]Скачивание изображений (256px CelebA-HQ)...[/cyan]")
    console.print("  Репозиторий: huggan/CelebA-HQ-256 (~2 GB)")

    try:
        from datasets import load_dataset
        from PIL import Image
        import io
        from tqdm import tqdm

        img_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("huggan/CelebA-HQ-256", split="train", streaming=True)

        saved = 0
        for sample in tqdm(ds, desc="Скачивание изображений"):
            img_raw = sample.get("image") or sample.get("img")
            if img_raw is None:
                continue
            if not isinstance(img_raw, Image.Image):
                img_raw = Image.open(io.BytesIO(img_raw)).convert("RGB")

            fname = f"{saved+1:06d}.jpg"
            img_raw.save(img_dir / fname, "JPEG", quality=95)
            saved += 1

        console.print(f"[green]✓ Скачано {saved:,} изображений (256px)[/green]")
        console.print("[yellow]Примечание: HF-версия содержит 30 000 изображений 256px "
                      "(вместо 202 599 × 178px). Для учебного проекта достаточно.[/yellow]")
        return True

    except Exception as e:
        console.print(f"[red]Ошибка HuggingFace datasets: {e}[/red]")
        console.print("[yellow]Попробуйте метод kaggle или ручное скачивание.[/yellow]")
        return False


def _download_annotations_hf(output_dir: Path):
    """Скачивает txt-файлы аннотаций через прямые ссылки HuggingFace."""
    import urllib.request

    files = {
        "list_attr_celeba.txt": (
            "https://huggingface.co/datasets/nateraw/celeb-a/"
            "resolve/main/list_attr_celeba.txt"
        ),
        "list_eval_partition.txt": (
            "https://huggingface.co/datasets/nateraw/celeb-a/"
            "resolve/main/list_eval_partition.txt"
        ),
    }
    for fname, url in files.items():
        dst = output_dir / fname
        if dst.exists():
            console.print(f"  [yellow]Уже есть: {fname}[/yellow]")
            continue
        console.print(f"  Скачиваем {fname}...")
        try:
            urllib.request.urlretrieve(url, str(dst))
            size_kb = dst.stat().st_size // 1024
            console.print(f"  [green]✓ {fname} ({size_kb:,} KB)[/green]")
        except Exception as e:
            console.print(f"  [yellow]Не удалось: {e}[/yellow]")


# ─── Инструкция для ручного скачивания ───────────────────────────────────────

def show_manual_instructions(output_dir: Path):
    console.print(Panel(
        "[bold]Ручное скачивание CelebA (работает всегда):[/bold]\n\n"
        "[bold cyan]Шаг 1 — Изображения (через браузер):[/bold cyan]\n"
        "Откройте ссылку в браузере и нажмите Download:\n"
        "  https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM\n"
        "  → Скачается img_align_celeba.zip (~1.3 GB)\n"
        f"  → Распакуйте в папку: {output_dir}\n\n"
        "[bold cyan]Шаг 2 — Аннотации атрибутов:[/bold cyan]\n"
        "  https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U\n"
        f"  → Поместите list_attr_celeba.txt в: {output_dir}\n\n"
        "[bold cyan]Шаг 3 — Разбивка train/val/test:[/bold cyan]\n"
        "  https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk\n"
        f"  → Поместите list_eval_partition.txt в: {output_dir}\n\n"
        f"[bold cyan]Итоговая структура {output_dir}/:[/bold cyan]\n"
        "  img_align_celeba/    ← 202 599 файлов .jpg\n"
        "  list_attr_celeba.txt\n"
        "  list_eval_partition.txt\n\n"
        "После скачивания запустите:\n"
        "  python stage1_data/download.py --check",
        title="Инструкция: ручное скачивание",
        border_style="blue",
    ))


# ─── Тестовый датасет ─────────────────────────────────────────────────────────

def create_test_dataset(output_dir: Path, n: int = 200):
    """
    Создаёт минимальный тестовый датасет (n синтетических изображений).
    Позволяет проверить весь пайплайн без скачивания реальных данных.

    ВНИМАНИЕ: для реального обучения нужен настоящий CelebA!
    """
    import numpy as np
    from PIL import Image

    # Импортируем CELEBA_ATTRS
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from stage1_data.make_prompts import CELEBA_ATTRS

    img_dir = output_dir / "img_align_celeba"
    img_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[yellow]Создание тестового датасета: {n} изображений...[/yellow]")

    filenames = []
    for i in range(n):
        fname = f"{i+1:06d}.jpg"
        # Цвет кожи со случайным оттенком
        base = (
            random.randint(170, 230),
            random.randint(130, 185),
            random.randint(90, 155),
        )
        arr = np.full((218, 178, 3), base, dtype=np.uint8)
        noise = np.random.randint(-15, 15, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / fname, "JPEG", quality=90)
        filenames.append(fname)

    # Атрибуты
    attr_path = output_dir / "list_attr_celeba.txt"
    with open(attr_path, "w") as f:
        f.write(f"{n}\n")
        f.write(" ".join(CELEBA_ATTRS) + "\n")
        for fname in filenames:
            vals = " ".join(str(random.choice([1, -1])) for _ in CELEBA_ATTRS)
            f.write(f"{fname} {vals}\n")

    # Разбивка
    partition_path = output_dir / "list_eval_partition.txt"
    with open(partition_path, "w") as f:
        for i, fname in enumerate(filenames):
            split = 0 if i < int(n*0.8) else (1 if i < int(n*0.9) else 2)
            f.write(f"{fname} {split}\n")

    console.print(f"[green]✓ Тестовый датасет создан: {n} изображений в {img_dir}[/green]")
    console.print("[bold yellow]⚠  ТЕСТОВЫЕ данные — только для проверки пайплайна![/bold yellow]")
    console.print("   Для реального обучения скачайте настоящий CelebA.")


# ─── Интерактивный выбор ──────────────────────────────────────────────────────

def interactive_menu(output_dir: Path) -> str:
    console.print(Panel(
        "Google Drive (gdown) заблокирован — слишком много запросов.\n"
        "Выберите альтернативный метод:",
        title="Ошибка: Google Drive недоступен",
        border_style="red",
    ))

    console.print("  [bold cyan][1][/bold cyan] Kaggle API [bold green](рекомендуется)[/bold green]"
                  " — бесплатно, надёжно, нужна регистрация")
    console.print("  [bold cyan][2][/bold cyan] HuggingFace datasets"
                  " — без регистрации, 30 000 изображений 256px")
    console.print("  [bold cyan][3][/bold cyan] Показать инструкцию для ручного скачивания")
    console.print("  [bold cyan][4][/bold cyan] Создать тестовые данные"
                  " [dim](200 изображений для проверки пайплайна)[/dim]")
    console.print()

    choice = input("Введите 1, 2, 3 или 4: ").strip()
    return {"1": "kaggle", "2": "huggingface", "3": "manual", "4": "test"}.get(choice, "kaggle")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Скачивание CelebA")
    parser.add_argument("--method",
                        choices=["kaggle", "huggingface", "manual", "test"],
                        default=None)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--check",  action="store_true",
                        help="Только проверить статус файлов")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Только проверка
    if args.check:
        console.print("[bold]Статус данных:[/bold]")
        check_existing(output_dir)
        return

    # Проверяем что уже есть
    console.print("[bold]Текущий статус:[/bold]")
    if check_existing(output_dir):
        console.print("\n[green]✓ Все файлы уже скачаны. Пропускаем.[/green]")
        return

    # Выбор метода
    method = args.method or interactive_menu(output_dir)
    console.print()

    success = False

    if method == "kaggle":
        success = download_via_kaggle(output_dir)

    elif method == "huggingface":
        success = download_via_huggingface(output_dir)

    elif method == "manual":
        show_manual_instructions(output_dir)
        sys.exit(0)

    elif method == "test":
        create_test_dataset(output_dir, n=200)
        success = True

    # Итог
    console.print("\n[bold]Итоговый статус:[/bold]")
    ok = check_existing(output_dir)

    if ok or success:
        console.print(Panel(
            "python stage1_data/make_prompts.py \\\n"
            "    --attrs    data/raw/list_attr_celeba.txt \\\n"
            "    --partition data/raw/list_eval_partition.txt \\\n"
            "    --images   data/raw/img_align_celeba \\\n"
            "    --output   data/processed",
            title="Следующий шаг — Этап 1.2",
            border_style="green",
        ))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
