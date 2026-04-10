#!/usr/bin/env python3
"""
stage4_eval/app.py
==================
Интерактивный Gradio-интерфейс для генерации портретов.

Ввод: чекбоксы атрибутов CelebA → промпт → генерация.

Запуск:
  pip install gradio
  python stage4_eval/app.py --lora lora_output

Откроется в браузере: http://localhost:7860
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1_data.make_prompts import attrs_to_prompt
from stage4_eval.generate import build_pipeline, generate_images


# ─── Группы атрибутов для UI ──────────────────────────────────────────────────

ATTR_GROUPS = {
    "Пол / Возраст": ["Male", "Young"],
    "Цвет волос":   ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
    "Стиль волос":  ["Bald", "Bangs", "Wavy_Hair", "Straight_Hair"],
    "Черты лица":   ["Big_Nose", "High_Cheekbones", "Narrow_Eyes",
                     "Chubby", "Double_Chin", "Pale_Skin", "Rosy_Cheeks"],
    "Растительность": ["Goatee", "Mustache", "5_o_Clock_Shadow", "Sideburns"],
    "Аксессуары":   ["Eyeglasses", "Wearing_Hat", "Wearing_Earrings",
                     "Wearing_Necklace", "Wearing_Necktie"],
    "Макияж":       ["Heavy_Makeup", "Wearing_Lipstick"],
    "Выражение":    ["Smiling", "Mouth_Slightly_Open"],
}

ATTR_LABELS = {
    "Male": "Мужчина", "Young": "Молодой/ая",
    "Black_Hair": "Чёрные волосы", "Blond_Hair": "Светлые волосы",
    "Brown_Hair": "Коричневые волосы", "Gray_Hair": "Седые волосы",
    "Bald": "Лысый", "Bangs": "Чёлка",
    "Wavy_Hair": "Волнистые волосы", "Straight_Hair": "Прямые волосы",
    "Big_Nose": "Большой нос", "High_Cheekbones": "Высокие скулы",
    "Narrow_Eyes": "Узкие глаза", "Chubby": "Пухлые щёки",
    "Double_Chin": "Двойной подбородок", "Pale_Skin": "Бледная кожа",
    "Rosy_Cheeks": "Румяные щёки",
    "Goatee": "Козлиная бородка", "Mustache": "Усы",
    "5_o_Clock_Shadow": "Щетина", "Sideburns": "Бакенбарды",
    "Eyeglasses": "Очки", "Wearing_Hat": "Шляпа",
    "Wearing_Earrings": "Серьги", "Wearing_Necklace": "Ожерелье",
    "Wearing_Necktie": "Галстук",
    "Heavy_Makeup": "Яркий макияж", "Wearing_Lipstick": "Помада",
    "Smiling": "Улыбка", "Mouth_Slightly_Open": "Рот приоткрыт",
}


def build_interface(
    lora_path: str,
    base_model: str = "runwayml/stable-diffusion-v1-5",
) -> gr.Blocks:
    """Строит Gradio-интерфейс."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"Загрузка модели ({device})...")
    pipe = build_pipeline(base_model, lora_path, device, dtype)
    print("✓ Модель готова")

    def generate(
        guidance_scale, num_steps, seed,
        # атрибуты — отдельные аргументы (Gradio не поддерживает dict)
        *attr_values
    ):
        # Собираем все атрибуты из позиционных аргументов
        all_attr_names = [a for group in ATTR_GROUPS.values() for a in group]
        attrs = {name: (1 if val else 0)
                 for name, val in zip(all_attr_names, attr_values)}

        prompt = attrs_to_prompt(attrs)

        images = generate_images(
            pipe, [prompt] * 4,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_steps),
            seed=int(seed) if seed else None,
        )
        return images, prompt

    with gr.Blocks(title="Portrait Generator — CelebA LoRA") as demo:
        gr.Markdown("# 🎭 Генератор портретов\nLoRA-дообучение Stable Diffusion на CelebA")

        with gr.Row():
            # Левая колонка — атрибуты
            with gr.Column(scale=1):
                gr.Markdown("### Атрибуты лица")
                attr_components = []

                for group_name, group_attrs in ATTR_GROUPS.items():
                    with gr.Group():
                        gr.Markdown(f"**{group_name}**")
                        for attr in group_attrs:
                            label = ATTR_LABELS.get(attr, attr)
                            default = attr in ("Young",)  # Young включён по умолчанию
                            cb = gr.Checkbox(label=label, value=default)
                            attr_components.append(cb)

                with gr.Accordion("Настройки генерации", open=False):
                    guidance = gr.Slider(1, 15, value=7.5, step=0.5,
                                         label="Guidance Scale")
                    steps    = gr.Slider(10, 100, value=50, step=5,
                                         label="Шагов DDIM")
                    seed_box = gr.Number(value=42, label="Seed (0 = случайно)",
                                         precision=0)

                btn = gr.Button("🎨 Генерировать", variant="primary")

            # Правая колонка — результаты
            with gr.Column(scale=2):
                prompt_box = gr.Textbox(label="Сгенерированный промпт",
                                         interactive=False)
                gallery    = gr.Gallery(label="Результаты",
                                         columns=2, rows=2,
                                         height=600)

        btn.click(
            fn=generate,
            inputs=[guidance, steps, seed_box] + attr_components,
            outputs=[gallery, prompt_box],
        )

        gr.Examples(
            examples=[
                [7.5, 50, 42, True, True] + [False] * (len(attr_components) - 2),
            ],
            inputs=[guidance, steps, seed_box] + attr_components,
            label="Примеры",
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio UI для генерации портретов")
    parser.add_argument("--lora",       default="lora_output")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true",
                        help="Создать публичную ссылку (Gradio share)")
    args = parser.parse_args()

    demo = build_interface(args.lora, args.base_model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
