#!/usr/bin/env python3
"""
run_all.py
==========
Мастер-скрипт: запускает все 4 этапа последовательно.

Использование:
  python run_all.py --stage 1          # только этап 1
  python run_all.py --stage 1 2 3 4    # все этапы
  python run_all.py --all              # все этапы
  python run_all.py --check            # проверить окружение
  python run_all.py --all --dry-run    # показать команды без запуска

Опции для быстрого теста (маленький датасет):
  python run_all.py --all --fast       # 1000 изображений, 500 шагов
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


COMMANDS = {
    1: {
        "name": "Данные и промпты",
        "desc": "Скачать CelebA, создать промпты, EDA",
        "steps": [
            ("Скачивание", "python stage1_data/download.py --method kaggle --output data/raw"),
            ("Промпты",    "python stage1_data/make_prompts.py "
                           "--attrs data/raw/list_attr_celeba.txt "
                           "--partition data/raw/list_eval_partition.txt "
                           "--images data/raw/img_align_celeba "
                           "--output data/processed"),
        ],
    },
    2: {
        "name": "Предобработка",
        "desc": "Ресайз до 512×512, обновить манифесты",
        "steps": [
            ("Предобработка", "python stage2_preprocess/preprocess.py "
                              "--input data/processed "
                              "--output data/processed "
                              "--size 512 --workers 4"),
            ("Проверка Dataset", "python stage2_preprocess/dataset.py "
                                  "data/processed/train_proc.json"),
        ],
    },
    3: {
        "name": "LoRA-обучение",
        "desc": "Дообучение Stable Diffusion 1.5",
        "steps": [
            ("Обучение", "python stage3_train/train_lora.py --config configs/config.yaml"),
        ],
    },
    4: {
        "name": "Инференс и оценка",
        "desc": "Генерация изображений, метрики FID/LPIPS/SSIM",
        "steps": [
            ("Генерация FID-датасета",
             "python stage4_eval/generate.py "
             "--lora lora_output "
             "--manifest data/processed/test_proc.json "
             "--n 1000 --batch-size 4 "
             "--output samples/fid_fake"),
            ("Копирование реальных изображений",
             "python -c \""
             "import shutil, json; from pathlib import Path; "
             "records=json.load(open('data/processed/test_proc.json'))[:1000]; "
             "out=Path('samples/fid_real'); out.mkdir(parents=True,exist_ok=True); "
             "[shutil.copy(r['image_path'], out/Path(r['image_path']).name) for r in records]"
             "\""),
            ("Оценка метрик",
             "python stage4_eval/metrics.py "
             "--real samples/fid_real "
             "--fake samples/fid_fake "
             "--prompts data/processed/test_proc.json "
             "--n 1000 --output evaluation_results.json"),
        ],
    },
}

FAST_OVERRIDES = {
    1: [("Промпты", " --max-train 1000")],
    2: [],
    3: [("Обучение", " --max-steps 500")],
    4: [("Генерация FID-датасета", " --n 200"),
        ("Копирование реальных изображений",
         "python -c \""
         "import shutil, json; from pathlib import Path; "
         "records=json.load(open('data/processed/test_proc.json'))[:200]; "
         "out=Path('samples/fid_real'); out.mkdir(parents=True,exist_ok=True); "
         "[shutil.copy(r['image_path'], out/Path(r['image_path']).name) for r in records]"
         "\""),
        ("Оценка метрик", " --n 200")],
}


def check_env() -> bool:
    """Проверяет, что все зависимости установлены."""
    checks = [
        ("torch",          "import torch; print(torch.__version__)"),
        ("diffusers",      "import diffusers; print(diffusers.__version__)"),
        ("transformers",   "import transformers; print(transformers.__version__)"),
        ("peft",           "import peft; print(peft.__version__)"),
        ("accelerate",     "import accelerate; print(accelerate.__version__)"),
        ("GPU",            "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"),
    ]

    table = Table(title="Проверка окружения")
    table.add_column("Компонент", style="cyan")
    table.add_column("Статус", style="green")
    table.add_column("Версия")

    all_ok = True
    for name, cmd in checks:
        result = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True, text=True
        )
        ok = result.returncode == 0
        if not ok:
            all_ok = False
        version = result.stdout.strip() if ok else result.stderr.strip()[:50]
        status  = "✓" if ok else "✗"
        color   = "green" if ok else "red"
        table.add_row(name, f"[{color}]{status}[/{color}]", version)

    console.print(table)
    return all_ok


def run_step(step_name: str, cmd: str, dry_run: bool = False) -> bool:
    """Запускает один шаг."""
    console.print(f"\n[bold cyan]→ {step_name}[/bold cyan]")
    console.print(f"  [dim]{cmd}[/dim]")

    if dry_run:
        return True

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        console.print(f"[red]✗ Ошибка в шаге '{step_name}' (код {result.returncode})[/red]")
        return False
    console.print(f"[green]✓ {step_name} завершён[/green]")
    return True


def run_stage(stage_num: int, dry_run: bool = False, fast: bool = False) -> bool:
    """Запускает один этап."""
    stage = COMMANDS[stage_num]
    overrides = FAST_OVERRIDES.get(stage_num, []) if fast else []

    console.print(Panel(
        f"Этап {stage_num}: {stage['name']}\n{stage['desc']}",
        border_style="blue",
    ))

    for step_name, cmd in stage["steps"]:
        # Применяем fast-override
        final_cmd = cmd
        for ovr_name, ovr_suffix in overrides:
            if ovr_name == step_name:
                # Если override — замена команды целиком
                if ovr_suffix.startswith("python "):
                    final_cmd = ovr_suffix
                else:
                    final_cmd = cmd + ovr_suffix

        ok = run_step(step_name, final_cmd, dry_run)
        if not ok:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Запуск этапов проекта")
    parser.add_argument("--stage",   nargs="+", type=int, default=None,
                        choices=[1, 2, 3, 4])
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--check",   action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать команды без запуска")
    parser.add_argument("--fast",    action="store_true",
                        help="Быстрый тест (маленький датасет, мало шагов)")
    args = parser.parse_args()

    if args.check:
        ok = check_env()
        sys.exit(0 if ok else 1)

    stages = list(range(1, 5)) if args.all else (args.stage or [])

    if not stages:
        console.print("[yellow]Укажите --stage 1 2 3 4 или --all[/yellow]")
        console.print("\nДоступные этапы:")
        for n, s in COMMANDS.items():
            console.print(f"  {n}: {s['name']} — {s['desc']}")
        sys.exit(0)

    if args.fast:
        console.print("[yellow]⚡ FAST MODE: уменьшенные данные и шаги[/yellow]")
    if args.dry_run:
        console.print("[yellow]🔍 DRY RUN: команды без запуска[/yellow]")

    for stage_num in stages:
        ok = run_stage(stage_num, dry_run=args.dry_run, fast=args.fast)
        if not ok:
            console.print(f"[red]✗ Этап {stage_num} завершился с ошибкой.[/red]")
            sys.exit(1)

    console.print(Panel(
        "Все этапы выполнены успешно!\n\n"
        "Результаты:\n"
        "  lora_output/pytorch_lora_weights.safetensors — LoRA-веса\n"
        "  evaluation_results.json — метрики\n"
        "  samples/ — примеры генерации\n\n"
        "Запустить UI:\n"
        "  python stage4_eval/app.py --lora lora_output",
        title="Готово!",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
