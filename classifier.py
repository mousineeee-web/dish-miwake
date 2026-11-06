from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_CACHE_DIR = Path(
    os.environ.get("DISH_MODEL_CACHE", Path(__file__).parent / ".model_cache")
)
MODEL_HUB_CACHE = MODEL_CACHE_DIR / "hub"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor


MODEL_NAME = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"

LABEL_KEYS = ("sushi", "ramen", "curry_rice", "fried_rice")
DISPLAY_LABELS = {
    "sushi": "寿司",
    "ramen": "ラーメン",
    "curry_rice": "カレーライス",
    "fried_rice": "チャーハン",
}
PROMPTS = {
    "English": {
        "sushi": "a photo of sushi",
        "ramen": "a photo of ramen",
        "curry_rice": "a photo of Japanese curry rice",
        "fried_rice": "a photo of fried rice",
    },
    "日本語": {
        "sushi": "寿司の写真",
        "ramen": "ラーメンの写真",
        "curry_rice": "カレーライスの写真",
        "fried_rice": "チャーハンの写真",
    },
}


@dataclass(frozen=True)
class Prediction:
    key: str
    label: str
    prompt: str
    score: float


@dataclass(frozen=True)
class PredictionResult:
    predictions: tuple[Prediction, ...]
    pixel_values_shape: tuple[int, ...]
    input_ids_shape: tuple[int, ...]
    raw_logits: tuple[float, ...]
    elapsed_seconds: float


def get_prompts(language: str) -> tuple[str, ...]:
    """Return prompts in the stable order defined by LABEL_KEYS."""
    if language not in PROMPTS:
        raise ValueError(f"Unsupported prompt language: {language}")
    return tuple(PROMPTS[language][key] for key in LABEL_KEYS)


def load_runtime():
    """Load the processor and model once. The model is kept outside Git."""
    MODEL_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=MODEL_HUB_CACHE,
        use_fast=False,
    )
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=MODEL_HUB_CACHE,
        use_safetensors=True,
    )
    model.eval()
    return processor, model


def predict(
    image: Image.Image,
    processor,
    model,
    language: str = "English",
) -> PredictionResult:
    """Classify one image against the four fixed dish prompts."""
    prompts = get_prompts(language)
    started = time.perf_counter()
    inputs = processor(
        text=list(prompts),
        images=image.convert("RGB"),
        padding="max_length",
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0].detach().cpu()
        scores = torch.sigmoid(logits)

    elapsed = time.perf_counter() - started
    predictions = tuple(
        sorted(
            (
                Prediction(
                    key=key,
                    label=DISPLAY_LABELS[key],
                    prompt=prompt,
                    score=float(score),
                )
                for key, prompt, score in zip(LABEL_KEYS, prompts, scores, strict=True)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
    )

    return PredictionResult(
        predictions=predictions,
        pixel_values_shape=tuple(inputs["pixel_values"].shape),
        input_ids_shape=tuple(inputs["input_ids"].shape),
        raw_logits=tuple(float(value) for value in logits),
        elapsed_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dish classification.")
    parser.add_argument("image", type=Path)
    parser.add_argument('--legacy-four', action='store_true', help='Reproduce the historical closed four-class baseline.')
    parser.add_argument(
        "--language",
        choices=tuple(PROMPTS),
        default="English",
        help="Language used for the candidate prompts.",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor, model = load_runtime()
    if args.legacy_four:
        result = predict(image, processor, model, args.language)
    else:
        from open_set import FoodRecognizer
        result = FoodRecognizer(processor, model).predict(image, args.language)
        print(f'mode: {result.catalog_version}')
        print(f'decision: {result.decision}')
        print(f'reason: {result.reason}')

    print(f"model: {MODEL_NAME}")
    print(f"image_mode: {image.mode}")
    print(f"image_size: {image.size}")
    print(f"pixel_values_shape: {result.pixel_values_shape}")
    print(f"input_ids_shape: {result.input_ids_shape}")
    print(f"raw_logits: {[round(value, 4) for value in result.raw_logits]}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.3f}")
    visible_predictions = result.predictions if args.legacy_four or result.decision == 'accepted' else ()
    for rank, item in enumerate(visible_predictions[:3], 1):
        print(f"{rank}: {item.label} ({item.prompt}) score={item.score:.4f}")


if __name__ == "__main__":
    main()
