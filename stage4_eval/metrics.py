#!/usr/bin/env python3
"""
stage4_eval/metrics.py
=======================
Вычисление всех метрик качества: FID, LPIPS, SSIM, CLIP-score.

Запуск (полная оценка):
  python stage4_eval/metrics.py \
      --real  data/processed/images/test \
      --fake  samples/fid_fake \
      --prompts data/processed/test_proc.json \
      --output evaluation_results.json

Запуск (только FID, быстро):
  python stage4_eval/metrics.py \
      --real data/processed/images/test \
      --fake samples/fid_fake \
      --metrics fid

Результаты сохраняются в evaluation_results.json и выводятся в консоль.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _image_paths(directory: Path, n: Optional[int] = None) -> List[Path]:
    """Возвращает отсортированный список путей к изображениям."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([p for p in directory.glob("*") if p.suffix.lower() in exts])
    if n:
        paths = paths[:n]
    if not paths:
        raise FileNotFoundError(f"Нет изображений в {directory}")
    return paths


def _load_pil(path: Path, size: int = 299) -> "np.ndarray":
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.array(img)


# ─── FID ──────────────────────────────────────────────────────────────────────

def compute_fid(
    real_dir: Path,
    fake_dir: Path,
    n_images: Optional[int] = None,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> float:
    """
    Вычисляет FID (Fréchet Inception Distance).

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))

    Чем ниже FID — тем ближе распределение сгенерированных изображений
    к реальным. FID < 10 — отличное качество.

    Использует pytorch-fid (pip install pytorch-fid).
    Для точного результата нужно ≥ 1000 изображений.
    """
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        log.error("Установите: pip install pytorch-fid")
        return -1.0

    # pytorch-fid принимает пути к папкам
    fid = calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )
    return float(fid)


# ─── LPIPS ────────────────────────────────────────────────────────────────────

def compute_lpips(
    real_paths: List[Path],
    fake_paths: List[Path],
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    net: str = "alex",
) -> float:
    """
    Вычисляет среднее LPIPS (Learned Perceptual Image Patch Similarity).

    LPIPS измеряет перцептивное сходство между парами изображений.
    Чем ниже — тем лучше.

    Использует lpips (pip install lpips).
    net='alex' — быстро, net='vgg' — ближе к восприятию человека.
    """
    try:
        import lpips
    except ImportError:
        log.error("Установите: pip install lpips")
        return -1.0

    from torchvision import transforms
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # → [-1,1] как ожидает lpips
    ])

    all_scores = []
    n = min(len(real_paths), len(fake_paths))

    for i in tqdm(range(0, n, batch_size), desc="LPIPS"):
        r_batch = torch.stack([
            transform(Image.open(p).convert("RGB"))
            for p in real_paths[i:i+batch_size]
        ]).to(device)
        f_batch = torch.stack([
            transform(Image.open(p).convert("RGB"))
            for p in fake_paths[i:i+batch_size]
        ]).to(device)

        with torch.no_grad():
            scores = loss_fn(r_batch, f_batch)  # (B, 1, 1, 1)
        all_scores.extend(scores.flatten().cpu().numpy().tolist())

    return float(np.mean(all_scores))


# ─── SSIM ─────────────────────────────────────────────────────────────────────

def compute_ssim(
    real_paths: List[Path],
    fake_paths: List[Path],
    size: int = 256,
) -> float:
    """
    Вычисляет среднее SSIM (Structural Similarity Index).

    SSIM ∈ [-1, 1], чем выше — тем лучше.
    Оценивает яркость, контраст и структуру изображения.

    Использует scikit-image.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        log.error("Установите: pip install scikit-image")
        return -1.0

    scores = []
    n = min(len(real_paths), len(fake_paths))

    for r_p, f_p in tqdm(zip(real_paths[:n], fake_paths[:n]),
                          total=n, desc="SSIM"):
        r = np.array(Image.open(r_p).convert("RGB").resize((size, size)))
        f = np.array(Image.open(f_p).convert("RGB").resize((size, size)))
        s = ssim(r, f, channel_axis=2, data_range=255)
        scores.append(s)

    return float(np.mean(scores))


# ─── CLIP-score ───────────────────────────────────────────────────────────────

def compute_clip_score(
    fake_paths: List[Path],
    prompts: List[str],
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """
    Вычисляет средний CLIP-score.

    CLIP-score = косинусное сходство между CLIP-эмбеддингом
    изображения и CLIP-эмбеддингом соответствующего текстового описания.

    Измеряет, насколько изображение соответствует промпту.
    Чем выше — тем лучше (0.28+ считается хорошим).

    Использует transformers CLIP.
    """
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(model_name)

    all_scores = []
    n = min(len(fake_paths), len(prompts))

    for i in tqdm(range(0, n, batch_size), desc="CLIP-score"):
        batch_imgs = [Image.open(p).convert("RGB") for p in fake_paths[i:i+batch_size]]
        batch_txt  = prompts[i:i+batch_size]

        inputs = proc(
            text=batch_txt,
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
        img_emb  = F.normalize(out.image_embeds, dim=-1)
        text_emb = F.normalize(out.text_embeds,  dim=-1)
        scores   = (img_emb * text_emb).sum(dim=-1)
        all_scores.extend(scores.cpu().numpy().tolist())

    return float(np.mean(all_scores))


# ─── Полная оценка ────────────────────────────────────────────────────────────

def evaluate_all(
    real_dir: Path,
    fake_dir: Path,
    prompts_manifest: Optional[Path] = None,
    n_images: int = 1000,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Запускает все метрики и возвращает сводный словарь.

    Args:
        real_dir: папка с реальными тестовыми изображениями
        fake_dir: папка со сгенерированными изображениями
        prompts_manifest: test_proc.json с промптами (для CLIP-score)
        n_images: сколько пар использовать
        metrics: список метрик (None = все: fid, lpips, ssim, clip_score)

    Returns:
        {"fid": 18.3, "lpips": 0.31, "ssim": 0.71, "clip_score": 0.29}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if metrics is None:
        metrics = ["fid", "lpips", "ssim", "clip_score"]

    real_paths = _image_paths(real_dir, n_images)
    fake_paths = _image_paths(fake_dir, n_images)
    n = min(len(real_paths), len(fake_paths))

    print(f"Real: {len(real_paths)} | Fake: {len(fake_paths)} | Оценка на {n}")
    print(f"Метрики: {metrics}\n")

    results = {}

    if "fid" in metrics:
        print("── FID ────────────────────────────────────────")
        fid = compute_fid(real_dir, fake_dir, n, batch_size, device)
        results["fid"] = round(fid, 3)
        print(f"FID = {fid:.3f}  (↓ лучше, цель ≤ 20)\n")

    if "lpips" in metrics:
        print("── LPIPS ──────────────────────────────────────")
        lp = compute_lpips(real_paths[:n], fake_paths[:n], batch_size, device)
        results["lpips"] = round(lp, 4)
        print(f"LPIPS = {lp:.4f}  (↓ лучше, цель ≤ 0.28)\n")

    if "ssim" in metrics:
        print("── SSIM ───────────────────────────────────────")
        ss = compute_ssim(real_paths[:n], fake_paths[:n])
        results["ssim"] = round(ss, 4)
        print(f"SSIM = {ss:.4f}  (↑ лучше, цель ≥ 0.72)\n")

    if "clip_score" in metrics and prompts_manifest:
        print("── CLIP-score ─────────────────────────────────")
        with open(prompts_manifest) as f:
            records = json.load(f)[:n]
        prompts = [r["prompt"] for r in records]
        cs = compute_clip_score(fake_paths[:n], prompts, batch_size, device)
        results["clip_score"] = round(cs, 4)
        print(f"CLIP-score = {cs:.4f}  (↑ лучше, цель ≥ 0.28)\n")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Оценка качества генерации")
    parser.add_argument("--real",    required=True,
                        help="Папка с реальными изображениями (тест)")
    parser.add_argument("--fake",    required=True,
                        help="Папка со сгенерированными")
    parser.add_argument("--prompts", default=None,
                        help="test_proc.json с промптами (для CLIP-score)")
    parser.add_argument("--n",       type=int, default=1000)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--metrics", nargs="+",
                        default=["fid", "lpips", "ssim", "clip_score"])
    parser.add_argument("--output",  default="evaluation_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = evaluate_all(
        real_dir=Path(args.real),
        fake_dir=Path(args.fake),
        prompts_manifest=Path(args.prompts) if args.prompts else None,
        n_images=args.n,
        batch_size=args.batch,
        device=device,
        metrics=args.metrics,
    )

    # Красивый вывод
    print("\n" + "="*50)
    print("  ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*50)
    targets = {"fid": ("≤20", "↓"), "lpips": ("≤0.28","↓"),
               "ssim": ("≥0.72","↑"), "clip_score": ("≥0.28","↑")}
    for k, v in results.items():
        tgt, arrow = targets.get(k, ("—",""))
        met = "✓" if (
            (arrow=="↓" and v <= float(tgt[1:])) or
            (arrow=="↑" and v >= float(tgt[1:]))
        ) else "✗"
        print(f"  {k:12s}: {v:8.4f}   цель {tgt} {met}")
    print("="*50)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены: {args.output}")


if __name__ == "__main__":
    main()
