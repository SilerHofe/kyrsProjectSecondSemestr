#!/usr/bin/env python3
"""
stage4_eval/generate.py
========================
Этап 4 — генерация изображений с обученным LoRA.

Использование:
  # По текстовому промпту
  python stage4_eval/generate.py \
      --lora lora_output \
      --prompt "A young woman with blonde hair, smiling" \
      --n 4 --output samples/

  # По атрибутам CelebA (JSON)
  python stage4_eval/generate.py \
      --lora lora_output \
      --attrs '{"Male":0,"Young":1,"Blond_Hair":1,"Smiling":1}' \
      --n 4

  # Генерация тестовой выборки для FID
  python stage4_eval/generate.py \
      --lora lora_output \
      --manifest data/processed/test_proc.json \
      --n-per-prompt 1 \
      --output samples/fid_fake \
      --batch-size 4

  # Без LoRA (базовая SD) — для сравнения
  python stage4_eval/generate.py \
      --no-lora \
      --prompt "A young woman with blonde hair, smiling" \
      --n 4 --output samples/baseline/
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1_data.make_prompts import attrs_to_prompt


# ─── Загрузка пайплайна ───────────────────────────────────────────────────────

def build_pipeline(
    base_model: str = "runwayml/stable-diffusion-v1-5",
    lora_path:  Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Строит StableDiffusionPipeline с (или без) LoRA.

    Args:
        base_model: HuggingFace model ID или локальный путь
        lora_path:  папка с LoRA-весами (None = базовая модель без LoRA)
        device:     "cuda" или "cpu"
        dtype:      float16 (GPU) или float32 (CPU)

    Returns:
        Готовый к инференсу pipeline

    Важно: для инференса используем DDIM-планировщик (50 шагов)
           вместо DDPM (1000 шагов) — в 20× быстрее при том же качестве.
    """
    print(f"Загрузка базовой модели: {base_model}")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,         # отключаем фильтр — лишний overhead
        requires_safety_checker=False,
    ).to(device)

    # DDIM для быстрого инференса
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Оптимизация памяти
    if device == "cuda":
        pipe.enable_attention_slicing()   # меньше VRAM, чуть медленнее

    # Загружаем LoRA
    if lora_path:
        lora_path = Path(lora_path)
        # Проверяем safetensors
        st_file = lora_path / "pytorch_lora_weights.safetensors"
        if st_file.exists():
            pipe.load_lora_weights(str(lora_path), weight_name="pytorch_lora_weights.safetensors")
            print(f"✓ LoRA загружена: {st_file}")
        else:
            # Пробуем загрузить через adapter_model.safetensors
            pipe.load_lora_weights(str(lora_path))
            print(f"✓ LoRA загружена: {lora_path}")
    else:
        print("⚠  LoRA не загружена — используется базовая SD 1.5")

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─── Генерация ────────────────────────────────────────────────────────────────

def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    num_images_per_prompt: int = 1,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    height: int = 512,
    width:  int = 512,
) -> List[Image.Image]:
    """
    Генерирует изображения по списку промптов.

    Args:
        prompts: список текстовых описаний
        num_images_per_prompt: сколько изображений на один промпт
        guidance_scale: сила влияния текста (7–8 рекомендуется)
        num_inference_steps: шагов DDIM (50 = качество, 20 = скорость)
        seed: для воспроизводимости (None = случайно)

    Returns:
        Список PIL Image

    Примечание о guidance_scale:
        w=3  → слабое следование тексту, разнообразные лица
        w=7.5→ баланс качества и разнообразия (рекомендуется)
        w=12 → точное следование тексту, иногда артефакты
    """
    generator = torch.manual_seed(seed) if seed is not None else None

    result = pipe(
        prompts,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    )
    return result.images


# ─── Генерация тестовой выборки для FID ──────────────────────────────────────

def generate_fid_dataset(
    pipe: StableDiffusionPipeline,
    manifest_path: Path,
    output_dir: Path,
    n_images: int = 1000,
    batch_size: int = 4,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
) -> None:
    """
    Генерирует N изображений по промптам из тестового манифеста.
    Используется для вычисления FID.

    Сохраняет изображения в output_dir/000000.png, 000001.png, ...

    Args:
        manifest_path: data/processed/test_proc.json
        output_dir: куда сохранять (samples/fid_fake)
        n_images: сколько генерировать (для FID нужно ≥ 1000)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        records = json.load(f)

    records = records[:n_images]
    prompts_all = [r["prompt"] for r in records]

    print(f"\nГенерация {len(prompts_all)} изображений для FID...")
    print(f"Сохраняем в: {output_dir}")

    saved = 0
    for i in tqdm(range(0, len(prompts_all), batch_size), desc="Генерация"):
        batch_prompts = prompts_all[i:i+batch_size]
        images = generate_images(
            pipe, batch_prompts,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        for img in images:
            img.save(output_dir / f"{saved:06d}.png")
            saved += 1

    print(f"✓ Сгенерировано и сохранено: {saved} изображений")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Генерация портретов с LoRA")
    parser.add_argument("--lora",         default="lora_output",
                        help="Путь к LoRA-весам")
    parser.add_argument("--no-lora",      action="store_true",
                        help="Запустить без LoRA (базовая SD 1.5)")
    parser.add_argument("--base-model",   default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt",       default=None)
    parser.add_argument("--attrs",        default=None,
                        help='JSON атрибутов, напр. \'{"Male":0,"Young":1}\'')
    parser.add_argument("--manifest",     default=None,
                        help="Манифест для генерации FID-датасета")
    parser.add_argument("--n",            type=int, default=4,
                        help="Число изображений (для --prompt / --attrs)")
    parser.add_argument("--n-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--guidance",     type=float, default=7.5)
    parser.add_argument("--steps",        type=int, default=50)
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument("--output",       default="samples/")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device == "cuda" else torch.float32

    lora_path = None if args.no_lora else args.lora

    pipe = build_pipeline(
        base_model=args.base_model,
        lora_path=lora_path,
        device=device,
        dtype=dtype,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Режим: генерация FID-датасета ─────────────────────────────────────
    if args.manifest:
        generate_fid_dataset(
            pipe=pipe,
            manifest_path=Path(args.manifest),
            output_dir=output_dir,
            n_images=args.n,
            batch_size=args.batch_size,
            guidance_scale=args.guidance,
            num_steps=args.steps,
        )
        return

    # ── Режим: генерация по атрибутам ─────────────────────────────────────
    if args.attrs:
        attrs = json.loads(args.attrs)
        prompt = attrs_to_prompt(attrs)
        print(f"Промпт из атрибутов: {prompt}")
    elif args.prompt:
        prompt = args.prompt
    else:
        # Дефолтный промпт для демонстрации
        prompt = ("A young woman with blonde wavy hair, smiling, "
                  "photorealistic portrait, high quality")
        print(f"Используем дефолтный промпт: {prompt}")

    # ── Генерация ──────────────────────────────────────────────────────────
    print(f"\nГенерация {args.n} изображений...")
    t0 = time.time()

    images = generate_images(
        pipe, [prompt] * args.n,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    dt = time.time() - t0
    print(f"Время: {dt:.1f}с ({dt/args.n:.1f}с на изображение)")

    for i, img in enumerate(images):
        path = output_dir / f"portrait_{i:03d}.png"
        img.save(path)
        print(f"  Сохранено: {path}")

    # Сетка
    if len(images) > 1:
        ncols = min(4, len(images))
        nrows = (len(images) + ncols - 1) // ncols
        w, h  = images[0].size
        grid  = Image.new("RGB", (w * ncols, h * nrows))
        for i, im in enumerate(images):
            r, c = divmod(i, ncols)
            grid.paste(im, (c * w, r * h))
        grid_path = output_dir / "grid.png"
        grid.save(grid_path)
        print(f"  Сетка: {grid_path}")


if __name__ == "__main__":
    main()
