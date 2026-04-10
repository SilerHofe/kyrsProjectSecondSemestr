#!/usr/bin/env python3
"""
stage3_train/train_lora.py
===========================
Этап 3 — LoRA-дообучение Stable Diffusion на CelebA.

Что происходит:
  1. Загружаем предобученную SD 1.5 (VAE + UNet + TextEncoder + Scheduler)
  2. Замораживаем ВСЁ кроме LoRA-адаптеров в UNet
  3. LoRA добавляет маленькие матрицы (r×d) к слоям cross-attention
  4. Обучаем только эти матрицы — ~3 MB вместо 860 MB
  5. Сохраняем LoRA-веса в lora_output/

Запуск (single GPU):
  python stage3_train/train_lora.py --config configs/config.yaml

Запуск (multi-GPU через accelerate):
  accelerate config
  accelerate launch stage3_train/train_lora.py --config configs/config.yaml

Запуск (быстрый тест, 100 шагов):
  python stage3_train/train_lora.py --config configs/config.yaml --max-steps 100

Время обучения (ориентир):
  RTX 3060 12GB, batch=4, grad_accum=4, 10 000 шагов → ~6–8 часов
  RTX 4090 24GB, batch=8, grad_accum=2, 10 000 шагов → ~2–3 часа
"""

import os
import sys
import math
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage2_preprocess.dataset import build_dataloaders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── Загрузка компонентов SD ─────────────────────────────────────────────────

def load_sd_components(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Загружает все компоненты Stable Diffusion 1.5.

    Компоненты:
        tokenizer    — CLIPTokenizer (77 токенов)
        text_encoder — CLIPTextModel (заморожен)
        vae          — AutoencoderKL (заморожен, сжатие 8×)
        unet         — UNet2DConditionModel (будет LoRA-обёртка)
        noise_sched  — DDPMScheduler (косинусный, T=1000)

    Важно: vae и text_encoder заморожены — они не обучаются!
    Только unet (через LoRA) изменяется при обучении.
    """
    log.info(f"Загрузка SD компонентов: {model_id}")

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # Замораживаем VAE и text_encoder — они не меняются
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    log.info(f"  UNet параметров: {sum(p.numel() for p in unet.parameters())/1e6:.0f}M")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


# ─── Применение LoRA к UNet ──────────────────────────────────────────────────

def apply_lora(unet: UNet2DConditionModel, rank: int, alpha: int,
               dropout: float, target_modules: list) -> UNet2DConditionModel:
    """
    Оборачивает UNet в LoRA с помощью PEFT.

    LoRA добавляет к каждому целевому слою (to_q, to_k, to_v, to_out.0)
    пару матриц:
        W_new = W_orig + (B × A) * (alpha / rank)

    где A ∈ R^(rank × d_in), B ∈ R^(d_out × rank) — обучаемые матрицы.
    W_orig заморожена, обновляются только A и B.

    Args:
        rank: размерность LoRA (4, 8, 16). Больше → точнее, медленнее.
        alpha: масштаб (обычно = 4 × rank или = rank)
        target_modules: слои для применения LoRA

    Returns:
        unet с LoRA-обёрткой
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


# ─── Один шаг обучения ───────────────────────────────────────────────────────

def training_step(
    batch: dict,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Вычисляет MSE-потерю для одного батча.

    Алгоритм:
        1. Кодируем изображения VAE → латенты z (4, 64, 64)
        2. Кодируем текст CLIP → context c (77, 768)
        3. Сэмплируем гауссовский шум ε ~ N(0, I)
        4. Случайный временной шаг t ~ U[0, T]
        5. Зашумляем: z_t = sqrt(ᾱ_t)*z + sqrt(1-ᾱ_t)*ε
        6. Предсказываем шум: ε_pred = UNet(z_t, t, c)
        7. Потеря: L = ||ε - ε_pred||²

    Returns:
        loss: скалярный тензор
    """
    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
    input_ids    = batch["input_ids"].to(device)

    # 1. Изображение → латент
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor   # ~0.18215

    # 2. Текст → эмбеддинг
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]  # (B, 77, 768)

    # 3. Шум и временные шаги
    noise = torch.randn_like(latents)
    B     = latents.shape[0]
    t     = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=device,
    ).long()

    # 4. Прямой процесс (замкнутая форма — один шаг!)
    noisy_latents = noise_scheduler.add_noise(latents, noise, t)

    # 5. Предсказание шума через UNet (с LoRA)
    noise_pred = unet(
        noisy_latents,
        t,
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    # 6. Потеря MSE
    return F.mse_loss(noise_pred, noise, reduction="mean")


# ─── Генерация валидационных примеров ────────────────────────────────────────

@torch.no_grad()
def generate_validation_images(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler,
    prompts: list,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float = 7.5,
    num_steps: int = 30,
    step: int = 0,
    output_dir: Path = Path("samples"),
) -> None:
    """
    Генерирует примеры изображений во время обучения.
    Использует DDIM для ускоренного инференса (30 шагов).
    """
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(num_steps)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for prompt in prompts:
        # Токенизация с условием и без (для CFG)
        def encode(text):
            ids = tokenizer(
                text, max_length=77, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            return text_encoder(ids)[0]

        ctx_cond   = encode(prompt)
        ctx_uncond = encode("")
        context    = torch.cat([ctx_uncond, ctx_cond])

        # Начальный шум
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        latents = latents * ddim.init_noise_sigma

        # DDIM-денойзинг
        for t in ddim.timesteps:
            lat_in = torch.cat([latents] * 2)
            lat_in = ddim.scale_model_input(lat_in, t)

            with torch.autocast(device.type, dtype=dtype):
                noise_pred = unet(
                    lat_in, t,
                    encoder_hidden_states=context,
                ).sample

            # Classifier-Free Guidance
            n_uncond, n_cond = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_cond - n_uncond)

            latents = ddim.step(noise_pred, t, latents).prev_sample

        # Декодирование
        latents = latents / vae.config.scaling_factor
        with torch.autocast(device.type, dtype=dtype):
            img = vae.decode(latents).sample
        img = (img.clamp(-1, 1) + 1) / 2   # → [0, 1]
        all_images.append(img)

    grid = torch.cat(all_images, dim=0)
    save_image(grid, output_dir / f"step_{step:07d}.png", nrow=len(prompts))
    log.info(f"Сохранены примеры: {output_dir}/step_{step:07d}.png")


# ─── Главный цикл обучения ────────────────────────────────────────────────────

def train(config: dict, max_steps_override: Optional[int] = None) -> None:
    """
    Полный цикл LoRA-обучения.

    Структура:
        for epoch in range(num_epochs):
            for batch in train_loader:
                with autocast():
                    loss = training_step(batch, ...) / grad_accum
                scaler.scale(loss).backward()

                if (step+1) % grad_accum == 0:
                    clip_grad_norm(lora_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % val_every == 0:
                    generate_validation_images(...)
                    save_lora_weights(...)
    """
    # Устройство и dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp     = config["training"]["mixed_precision"]
    dtype  = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float32)
    log.info(f"Устройство: {device} | dtype: {dtype}")

    # Директории
    output_dir = Path(config["paths"]["lora_output"])
    samples_dir = Path(config["paths"]["samples"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Компоненты модели ──────────────────────────────────────────────────
    model_id = config["lora"]["base_model"]
    tokenizer, text_encoder, vae, unet, noise_scheduler = \
        load_sd_components(model_id, device, dtype)

    # ── Применяем LoRA ─────────────────────────────────────────────────────
    unet = apply_lora(
        unet,
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
    )

    # ── Данные ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_manifest=config["paths"]["train_manifest"].replace(".json", "_proc.json"),
        val_manifest=config["paths"]["val_manifest"].replace(".json", "_proc.json"),
        batch_size=config["training"]["train_batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        tokenizer_name=config["clip"]["model_name"],
        max_train_samples=config["data"].get("max_train_samples"),
    )

    # ── Оптимизатор ───────────────────────────────────────────────────────
    # Только LoRA-параметры (requires_grad=True)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    log.info(f"Обучаемых параметров: {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            lora_params,
            lr=config["training"]["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        log.info("Используем 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=config["training"]["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        log.info("Используем стандартный AdamW (bitsandbytes не установлен)")

    # ── Планировщик LR ────────────────────────────────────────────────────
    grad_accum  = config["training"]["gradient_accumulation_steps"]
    num_epochs  = config["training"]["num_train_epochs"]
    num_updates = math.ceil(len(train_loader) * num_epochs / grad_accum)

    if max_steps_override:
        num_updates = max_steps_override

    lr_scheduler = get_scheduler(
        name=config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=num_updates,
    )

    # ── Scaler для mixed precision ────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(mp == "fp16"))

    # ── Промпты для валидации ─────────────────────────────────────────────
    val_prompts = [
        "A young woman with blonde wavy hair, smiling, photorealistic portrait, high quality",
        "An elderly man with gray hair and mustache, wearing glasses, photorealistic portrait",
        "A middle-aged woman with brown hair, heavy makeup, high cheekbones, portrait",
        "A young man with black hair, no beard, photorealistic portrait, high quality",
    ]

    # ── Цикл обучения ─────────────────────────────────────────────────────
    log.info(f"Начало обучения: {num_updates} шагов оптимизатора")
    log.info(f"Эпох: {num_epochs} | Батч: {config['training']['train_batch_size']} "
             f"× grad_accum {grad_accum} = {config['training']['train_batch_size']*grad_accum}")

    global_step = 0
    optimizer.zero_grad()

    save_every = config["training"]["save_every_n_steps"]
    val_every  = config["training"]["validation_every_n_steps"]

    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}")

        for step_in_epoch, batch in enumerate(pbar):
            # Выход если достигнут max_steps_override
            if max_steps_override and global_step >= max_steps_override:
                break

            # Прямой + обратный проход
            with torch.autocast(device.type, dtype=dtype, enabled=(mp != "no")):
                loss = training_step(
                    batch, unet, vae, text_encoder,
                    noise_scheduler, device, dtype,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum

            # Шаг оптимизатора каждые grad_accum итераций
            if (step_in_epoch + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{epoch_loss / (step_in_epoch+1):.4f}",
                    "lr":   f"{lr:.2e}",
                    "step": global_step,
                })

                # Валидация и сохранение примеров
                if global_step % val_every == 0:
                    unet.eval()
                    generate_validation_images(
                        unet, vae, text_encoder, tokenizer,
                        noise_scheduler, val_prompts,
                        device, dtype, step=global_step,
                        output_dir=samples_dir,
                    )
                    unet.train()

                # Сохранение чекпоинта
                if global_step % save_every == 0:
                    _save_lora(unet, output_dir, global_step)

        avg_loss = epoch_loss / len(train_loader)
        log.info(f"Эпоха {epoch+1} завершена | avg_loss={avg_loss:.4f}")

    # Финальное сохранение
    _save_lora(unet, output_dir, global_step, final=True)
    log.info("Обучение завершено!")
    log.info(f"LoRA-веса: {output_dir}/pytorch_lora_weights.safetensors")


def _save_lora(unet, output_dir: Path, step: int, final: bool = False) -> None:
    """Сохраняет LoRA-веса."""
    # Чекпоинт шага
    ckpt_dir = output_dir / f"checkpoint-{step}"
    unet.save_pretrained(ckpt_dir)
    log.info(f"Чекпоинт сохранён: {ckpt_dir}")

    if final:
        # Финальные веса в корень output_dir
        unet.save_pretrained(output_dir)
        # Также сохраняем в safetensors формате
        try:
            from peft.utils import get_peft_model_state_dict
            from safetensors.torch import save_file
            state_dict = get_peft_model_state_dict(unet)
            save_file(state_dict, output_dir / "pytorch_lora_weights.safetensors")
            log.info("Финальные веса сохранены в safetensors")
        except Exception as e:
            log.warning(f"Не удалось сохранить safetensors: {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA-дообучение SD на CelebA")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Ограничить число шагов (для тестирования)")
    parser.add_argument("--rank",      type=int, default=None,
                        help="Переопределить LoRA rank")
    parser.add_argument("--lr",        type=float, default=None,
                        help="Переопределить learning rate")
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # Переопределения из CLI
    if args.rank: config["lora"]["rank"] = args.rank
    if args.lr:   config["training"]["learning_rate"] = args.lr

    train(config, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
