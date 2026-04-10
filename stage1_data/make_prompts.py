#!/usr/bin/env python3
"""
stage1_data/make_prompts.py
============================
Этап 1 — главный скрипт.

Читает list_attr_celeba.txt и list_eval_partition.txt,
формирует текстовые промпты для каждого изображения,
сохраняет манифесты train/val/test.json.

Запуск:
  python stage1_data/make_prompts.py \
      --attrs    data/raw/list_attr_celeba.txt \
      --partition data/raw/list_eval_partition.txt \
      --images   data/raw/img_align_celeba \
      --output   data/processed

Выход:
  data/processed/
      train.json  — [{filename, image_path, attrs, prompt}, ...]
      val.json
      test.json
      celeba_prompts.json  — все 202 599 записей
      stats.json           — статистика атрибутов
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()


# ─── 40 атрибутов CelebA ──────────────────────────────────────────────────────

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young",
]

# Атрибуты, которые используются в промпте
PROMPT_ATTRS = {
    # пол / возраст
    "Male", "Young",
    # волосы
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair",
    "Bald", "Bangs", "Wavy_Hair", "Straight_Hair", "Receding_Hairline",
    # лицо
    "Big_Nose", "Pointy_Nose", "High_Cheekbones", "Oval_Face",
    "Chubby", "Double_Chin", "Narrow_Eyes", "Pale_Skin",
    # растительность (только для мужчин)
    "5_o_Clock_Shadow", "Goatee", "Mustache", "Sideburns", "No_Beard",
    # аксессуары
    "Eyeglasses", "Wearing_Hat", "Wearing_Earrings",
    "Wearing_Necklace", "Wearing_Necktie",
    # макияж
    "Heavy_Makeup", "Wearing_Lipstick", "Rosy_Cheeks",
    # выражение
    "Smiling", "Mouth_Slightly_Open",
}


# ─── Шаблонизатор промптов ────────────────────────────────────────────────────

def attrs_to_prompt(attrs: Dict[str, int]) -> str:
    """
    Преобразует вектор бинарных атрибутов CelebA в текстовое описание.

    Структура промпта:
        "A {age} {gender} with {hair}, {features}, {accessories},
         {expression}, photorealistic portrait, high quality"

    Args:
        attrs: словарь {attr_name: 0/1}

    Returns:
        Текстовый промпт на английском

    Примеры:
        → "A young woman with blonde wavy hair, smiling, wearing earrings,
           photorealistic portrait, high quality"
        → "An elderly man with gray hair and goatee, wearing glasses,
           photorealistic portrait, high quality"
    """
    parts = []

    # ── Пол и возраст ──────────────────────────────────────────────────
    is_male  = attrs.get("Male", 0) == 1
    is_young = attrs.get("Young", 0) == 1
    gender   = "man" if is_male else "woman"

    if is_young:
        age = "young"
        article = "A"
    else:
        age = "middle-aged"
        article = "A"
        # Немного разнообразия для пожилых
        if attrs.get("Gray_Hair", 0) or attrs.get("Receding_Hairline", 0):
            age = "elderly"
            article = "An"

    parts.append(f"{article} {age} {gender}")

    # ── Волосы ─────────────────────────────────────────────────────────
    hair_parts = []

    if attrs.get("Bald", 0):
        hair_parts.append("bald")
    else:
        # Цвет
        color = None
        for c, label in [("Black_Hair", "black"), ("Blond_Hair", "blonde"),
                          ("Brown_Hair", "brown"), ("Gray_Hair", "gray")]:
            if attrs.get(c, 0):
                color = label
                break

        # Стиль
        style = None
        if attrs.get("Wavy_Hair", 0):
            style = "wavy"
        elif attrs.get("Straight_Hair", 0):
            style = "straight"
        elif attrs.get("Bangs", 0):
            style = "with bangs"

        if color and style:
            hair_parts.append(f"{color} {style} hair")
        elif color:
            hair_parts.append(f"{color} hair")
        elif style:
            hair_parts.append(f"{style} hair")
        else:
            hair_parts.append("hair")  # общее упоминание

        if attrs.get("Receding_Hairline", 0):
            hair_parts.append("receding hairline")

    if hair_parts:
        parts.append("with " + " and ".join(hair_parts))

    # ── Черты лица ─────────────────────────────────────────────────────
    face_features = []
    if attrs.get("Big_Nose", 0):
        face_features.append("a prominent nose")
    if attrs.get("High_Cheekbones", 0):
        face_features.append("high cheekbones")
    if attrs.get("Chubby", 0):
        face_features.append("chubby cheeks")
    if attrs.get("Double_Chin", 0):
        face_features.append("a double chin")
    if attrs.get("Narrow_Eyes", 0):
        face_features.append("narrow eyes")
    if attrs.get("Pale_Skin", 0):
        face_features.append("pale skin")
    if attrs.get("Rosy_Cheeks", 0):
        face_features.append("rosy cheeks")
    if face_features:
        parts.append(", ".join(face_features))

    # ── Растительность (только мужчины) ────────────────────────────────
    if is_male:
        beard_parts = []
        if attrs.get("Goatee", 0):
            beard_parts.append("a goatee")
        elif attrs.get("Mustache", 0):
            beard_parts.append("a mustache")
        elif attrs.get("5_o_Clock_Shadow", 0):
            beard_parts.append("stubble")
        elif attrs.get("Sideburns", 0):
            beard_parts.append("sideburns")
        if beard_parts:
            parts.append("with " + ", ".join(beard_parts))

    # ── Аксессуары ─────────────────────────────────────────────────────
    accessories = []
    if attrs.get("Eyeglasses", 0):
        accessories.append("wearing glasses")
    if attrs.get("Wearing_Hat", 0):
        accessories.append("wearing a hat")
    if attrs.get("Wearing_Necktie", 0):
        accessories.append("wearing a tie")
    if attrs.get("Wearing_Necklace", 0):
        accessories.append("wearing a necklace")
    if attrs.get("Wearing_Earrings", 0) and not is_male:
        accessories.append("wearing earrings")
    if accessories:
        parts.append(", ".join(accessories))

    # ── Макияж (только женщины) ────────────────────────────────────────
    if not is_male:
        makeup = []
        if attrs.get("Heavy_Makeup", 0):
            makeup.append("with heavy makeup")
        if attrs.get("Wearing_Lipstick", 0) and not attrs.get("Heavy_Makeup", 0):
            makeup.append("wearing lipstick")
        if makeup:
            parts.append(", ".join(makeup))

    # ── Выражение лица ─────────────────────────────────────────────────
    expression = []
    if attrs.get("Smiling", 0):
        expression.append("smiling")
    elif attrs.get("Mouth_Slightly_Open", 0):
        expression.append("with mouth slightly open")
    if expression:
        parts.append(", ".join(expression))

    # ── Суффикс для качества (важно для SD!) ───────────────────────────
    suffix = "photorealistic portrait, high quality, detailed"

    prompt = " ".join(parts) + ", " + suffix
    return prompt


# ─── Парсинг файла атрибутов ──────────────────────────────────────────────────

def _detect_format(first_line: str) -> str:
    """Определяет формат файла: 'csv' или 'classic'."""
    # CSV: первая строка содержит запятые и слова (заголовок)
    if "," in first_line and not first_line.strip().isdigit():
        return "csv"
    # Classic: первая строка — число изображений
    return "classic"


def parse_attr_file(attr_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Читает файл атрибутов CelebA.

    Поддерживает два формата автоматически:

    Формат 1 — классический (оригинальный CelebA .txt):
        202599
        5_o_Clock_Shadow Arched_Eyebrows ...
        000001.jpg -1  1  1 ...

    Формат 2 — CSV (Kaggle-версия):
        image_id,5_o_Clock_Shadow,Arched_Eyebrows,...
        000001.jpg,-1,1,1,...

    Returns:
        {filename: {attr_name: 0/1}}
    """
    console.print(f"[cyan]Читаем атрибуты из {attr_path}...[/cyan]")

    attrs = {}

    with open(attr_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        fmt = _detect_format(first_line)
        console.print(f"  Формат файла: [yellow]{fmt}[/yellow]")

        if fmt == "csv":
            # ── CSV формат ──────────────────────────────────────────────
            import csv, io

            # Первая строка уже прочитана — это заголовок
            header = [h.strip() for h in first_line.strip().split(",")]

            # Колонка с именем файла: обычно 'image_id' или первая колонка
            id_col = header[0]  # 'image_id'
            attr_names = header[1:]  # остальные — атрибуты

            reader = csv.DictReader(f, fieldnames=header)
            for row in tqdm(reader, desc="Парсинг CSV атрибутов"):
                filename = row[id_col].strip()
                # Нормализуем имя: добавляем .jpg если нет расширения
                if not filename.endswith(".jpg"):
                    filename = filename + ".jpg"
                values = {}
                for name in attr_names:
                    raw = row.get(name, "0").strip()
                    try:
                        values[name] = 1 if int(raw) == 1 else 0
                    except ValueError:
                        values[name] = 0
                attrs[filename] = values

        else:
            # ── Классический формат ─────────────────────────────────────
            # first_line уже прочитана — это число изображений
            try:
                n_images = int(first_line.strip())
            except ValueError:
                # Иногда число отсутствует — читаем заново
                n_images = None

            attr_names = f.readline().strip().split()

            for line in tqdm(f, total=n_images, desc="Парсинг атрибутов"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename = parts[0]
                values = {
                    name: (1 if int(v) == 1 else 0)
                    for name, v in zip(attr_names, parts[1:])
                }
                attrs[filename] = values

    console.print(
        f"[green]✓ Загружено: {len(attrs):,} записей, "
        f"{len(next(iter(attrs.values()))) if attrs else 0} атрибутов[/green]"
    )
    return attrs


def parse_partition_file(partition_path: Path) -> Dict[str, int]:
    """
    Читает файл разбивки train/val/test.

    Поддерживает два формата автоматически:

    Формат 1 — классический:
        000001.jpg 0
        000002.jpg 1

    Формат 2 — CSV (Kaggle):
        image_id,partition
        000001.jpg,0
        000002.jpg,1
    """
    partition = {}

    with open(partition_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        fmt = _detect_format(first_line)

        if fmt == "csv":
            # CSV: пропускаем заголовок (уже прочитан), читаем остальное
            import csv
            header = [h.strip() for h in first_line.strip().split(",")]
            id_col   = header[0]       # 'image_id'
            part_col = header[-1]      # 'partition' или последняя колонка

            reader = csv.DictReader(f, fieldnames=header)
            for row in reader:
                filename = row[id_col].strip()
                if not filename.endswith(".jpg"):
                    filename = filename + ".jpg"
                try:
                    partition[filename] = int(row[part_col].strip())
                except (ValueError, KeyError):
                    pass
        else:
            # Классический: first_line — первая запись (уже прочитана)
            for line in [first_line] + f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        partition[parts[0]] = int(parts[1])
                    except ValueError:
                        pass

    console.print(f"[green]✓ Разбивка: {len(partition):,} записей[/green]")
    return partition


# ─── EDA (Exploratory Data Analysis) ─────────────────────────────────────────

def print_eda(attrs_map: Dict[str, Dict[str, int]]) -> None:
    """Выводит статистику распределения атрибутов."""
    n_total = len(attrs_map)

    counts = defaultdict(int)
    for rec in attrs_map.values():
        for attr, val in rec.items():
            if val == 1:
                counts[attr] += 1

    table = Table(title=f"Распределение атрибутов CelebA (N={n_total:,})")
    table.add_column("Атрибут", style="cyan", width=25)
    table.add_column("Кол-во", style="green", justify="right")
    table.add_column("% от датасета", style="yellow", justify="right")
    table.add_column("Предупреждение", style="red")

    for attr in sorted(CELEBA_ATTRS):
        cnt  = counts[attr]
        pct  = cnt / n_total * 100
        warn = "⚠ Редкий (<5%)" if pct < 5 else ""
        table.add_row(attr, f"{cnt:,}", f"{pct:.1f}%", warn)

    console.print(table)


def compute_attr_stats(attrs_map: Dict[str, Dict[str, int]]) -> Dict:
    """Вычисляет статистику для stats.json."""
    n = len(attrs_map)
    stats = {"n_total": n, "attributes": {}}
    for attr in CELEBA_ATTRS:
        cnt = sum(1 for rec in attrs_map.values() if rec.get(attr, 0) == 1)
        stats["attributes"][attr] = {
            "count":   cnt,
            "percent": round(cnt / n * 100, 2),
            "rare":    cnt / n < 0.05,
        }
    return stats


# ─── Основная функция ─────────────────────────────────────────────────────────

def make_prompts(
    attr_path: Path,
    partition_path: Optional[Path],
    images_dir: Path,
    output_dir: Path,
    show_eda: bool = True,
    max_samples: Optional[int] = None,
) -> None:
    """
    Полный пайплайн создания промптов и манифестов.

    Выходные файлы:
        output_dir/train.json      — список записей обучающей выборки
        output_dir/val.json        — валидационная
        output_dir/test.json       — тестовая (НЕ использовать при обучении!)
        output_dir/celeba_prompts.json  — все записи
        output_dir/stats.json      — статистика атрибутов
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Атрибуты
    attrs_map = parse_attr_file(attr_path)

    # 2. EDA
    if show_eda:
        print_eda(attrs_map)

    # 3. Статистика
    stats = compute_attr_stats(attrs_map)
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 4. Разбивка
    if partition_path and partition_path.exists():
        partition = parse_partition_file(partition_path)
    else:
        console.print("[yellow]Файл разбивки не найден — используем random 90/5/5[/yellow]")
        import random
        keys = list(attrs_map.keys())
        random.seed(42)
        random.shuffle(keys)
        n = len(keys)
        partition = {}
        for i, k in enumerate(keys):
            if i < int(n * 0.90):
                partition[k] = 0
            elif i < int(n * 0.95):
                partition[k] = 1
            else:
                partition[k] = 2

    # 5. Генерация промптов и сборка записей
    all_records  = []
    split_counts = {0: 0, 1: 0, 2: 0}
    missing_imgs = 0

    console.print("[cyan]Генерация промптов...[/cyan]")

    for filename, attrs in tqdm(attrs_map.items(), desc="Промпты"):
        img_path = images_dir / filename
        if not img_path.exists():
            missing_imgs += 1
            continue

        prompt = attrs_to_prompt(attrs)
        split  = partition.get(filename, 0)

        record = {
            "filename":   filename,
            "image_path": str(img_path),
            "split":      split,       # 0=train, 1=val, 2=test
            "attrs":      attrs,
            "prompt":     prompt,
        }
        all_records.append(record)
        split_counts[split] += 1

    console.print(
        f"[green]✓ Записей: {len(all_records):,} "
        f"(пропущено {missing_imgs} без изображения)[/green]"
    )
    console.print(
        f"  train: {split_counts[0]:,} | "
        f"val: {split_counts[1]:,} | "
        f"test: {split_counts[2]:,}"
    )

    # 6. Сохранение
    # Все
    with open(output_dir / "celeba_prompts.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Разбивка
    split_names = {0: "train", 1: "val", 2: "test"}
    for split_id, split_name in split_names.items():
        split_records = [r for r in all_records if r["split"] == split_id]
        if max_samples and split_id == 0:
            split_records = split_records[:max_samples]
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_records, f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓ {split_name}.json → {len(split_records):,} записей[/green]")

    # 7. Примеры промптов
    console.print("\n[bold cyan]Примеры сгенерированных промптов:[/bold cyan]")
    import random
    for rec in random.sample(all_records, min(5, len(all_records))):
        console.print(f"  [dim]{rec['filename']}[/dim]")
        console.print(f"  → [green]{rec['prompt']}[/green]\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_path(given: str, alternatives: list) -> Path:
    """
    Возвращает первый существующий путь из списка.
    Если ни один не найден — возвращает исходный (ошибка обнаружится позже).
    """
    p = Path(given)
    if p.exists():
        return p
    for alt in alternatives:
        ap = Path(alt)
        if ap.exists():
            console.print(f"[yellow]Найден альтернативный файл: {ap}[/yellow]")
            return ap
    return p  # вернём оригинал, ошибка будет понятной


def main():
    parser = argparse.ArgumentParser(description="Создание промптов для CelebA")
    parser.add_argument("--attrs",     default="data/raw/list_attr_celeba.txt")
    parser.add_argument("--partition", default="data/raw/list_eval_partition.txt")
    parser.add_argument("--images",    default="data/raw/img_align_celeba")
    parser.add_argument("--output",    default="data/processed")
    parser.add_argument("--eda",       action="store_true", default=True,
                        help="Показать статистику атрибутов")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Ограничить train выборку (для отладки)")
    args = parser.parse_args()

    # Автоопределение файлов — поддерживаем и .txt и .csv
    attr_path = _resolve_path(args.attrs, [
        "data/raw/list_attr_celeba.csv",
        "data/raw/celeba-dataset/list_attr_celeba.txt",
        "data/raw/celeba-dataset/list_attr_celeba.csv",
    ])
    partition_path = _resolve_path(args.partition, [
        "data/raw/list_eval_partition.csv",
        "data/raw/celeba-dataset/list_eval_partition.txt",
        "data/raw/celeba-dataset/list_eval_partition.csv",
    ])

    console.print(f"  Атрибуты:  [cyan]{attr_path}[/cyan]")
    console.print(f"  Разбивка:  [cyan]{partition_path}[/cyan]")
    console.print(f"  Изображения: [cyan]{args.images}[/cyan]")

    make_prompts(
        attr_path=attr_path,
        partition_path=partition_path,
        images_dir=Path(args.images),
        output_dir=Path(args.output),
        show_eda=args.eda,
        max_samples=args.max_train,
    )

    console.print(Panel(
        "Следующий шаг:\n"
        "  python stage2_preprocess/preprocess.py --config configs/config.yaml",
        title="Этап 1 завершён!",
        border_style="green",
    ))


if __name__ == "__main__":
    main()

