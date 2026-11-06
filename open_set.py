"""Conservative open-set dish suggestions using the existing local SigLIP."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageStat

from classifier import Prediction
from food_catalog import CATALOG_VERSION, DISHES, FOOD_PROMPTS, OTHER_FOOD_PROMPTS, NON_FOOD_PROMPTS, all_prompts


CONFIG_PATH = Path(__file__).with_name('open_set_config.json')
CONFIG = json.loads(CONFIG_PATH.read_text())


@dataclass(frozen=True)
class FoodResult:
    decision: str
    reason: str
    predictions: tuple[Prediction, ...]
    diagnostics: dict
    elapsed_seconds: float
    pixel_values_shape: tuple[int, ...] = ()
    input_ids_shape: tuple[int, ...] = ()
    raw_logits: tuple[float, ...] = ()
    catalog_version: str = CATALOG_VERSION


def decide(logits, config=None):
    """Never force a dish name when negative/other evidence wins or ties."""
    config = CONFIG if config is None else config
    expected = len(DISHES) + len(FOOD_PROMPTS) + len(OTHER_FOOD_PROMPTS) + len(NON_FOOD_PROMPTS)
    if len(logits) != expected or not all(math.isfinite(float(x)) for x in logits):
        raise ValueError('Invalid model scores')
    known = list(map(float, logits[:len(DISHES)]))
    generic_start = len(DISHES)
    other_start = generic_start + len(FOOD_PROMPTS)
    nonfood_start = other_start + len(OTHER_FOOD_PROMPTS)
    ranked = sorted(range(len(known)), key=lambda i: known[i], reverse=True)
    best, second = known[ranked[0]], known[ranked[1]]
    generic = max(logits[generic_start:other_start])
    other = max(logits[other_start:nonfood_start])
    nonfood = max(logits[nonfood_start:])
    food_margin = max(best, generic, other) - nonfood
    diagnostics = {'food_margin': float(food_margin), 'dish_margin': best-second,
                   'other_margin': best-float(other), 'generic_gap': float(generic)-best,
                   'best_dish_logit': best, 'best_nonfood_logit': float(nonfood),
                   'best_other_food_logit': float(other)}
    if food_margin <= config['min_food_margin']:
        return 'non_food', '食べ物・飲み物の写真と確認できませんでした。対象以外の画像の可能性があります。', diagnostics
    if best - other <= config['min_other_margin']:
        return 'out_of_scope', '対象外の食品の可能性があるため、名前を確定しません。', diagnostics
    if generic - best > config['max_generic_gap']:
        return 'out_of_scope', f'食品の可能性はありますが、対象{len(DISHES)}種類では判断できませんでした。', diagnostics
    if best - second <= config['min_dish_margin']:
        return 'uncertain', '複数の候補が近いため、名前を確定できませんでした。', diagnostics
    return 'accepted', f'対象{len(DISHES)}種類の中で比較した参考候補です。正解を保証するものではありません。', diagnostics


class FoodRecognizer:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model
        self._text_cache = {}
        self._lock = threading.Lock()

    def _text_features(self, language):
        with self._lock:
            if language not in self._text_cache:
                tokens = self.processor(text=list(all_prompts(language)), padding='max_length', return_tensors='pt')
                with torch.inference_mode():
                    features = self.model.get_text_features(**tokens)
                    features = torch.nn.functional.normalize(features, dim=-1)
                self._text_cache[language] = features, tuple(tokens['input_ids'].shape)
            return self._text_cache[language]

    def predict(self, image: Image.Image, language='English'):
        prompts = all_prompts(language)  # Validate before loading anything.
        started = time.perf_counter()
        image = image.convert('RGB')
        if max(ImageStat.Stat(image.resize((64, 64))).stddev) <= CONFIG['flat_image_stddev']:
            return FoodResult('insufficient_image', '単色に近い画像のため、料理を判断できません。', (), {}, time.perf_counter()-started)
        with torch.inference_mode():
            text_features, token_shape = self._text_features(language)
            inputs = self.processor(images=image, return_tensors='pt')
            image_features = self.model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            logits = (image_features @ text_features.T * self.model.logit_scale.exp() + self.model.logit_bias)[0].cpu()
        decision, reason, diagnostics = decide(logits.tolist())
        scores = torch.sigmoid(logits[:len(DISHES)])
        predictions = tuple(sorted((Prediction(key, label, prompt, float(score))
                                    for (key, label, _, _), prompt, score in zip(DISHES, prompts, scores)),
                                   key=lambda item: item.score, reverse=True))
        return FoodResult(decision, reason, predictions, diagnostics, time.perf_counter()-started,
                          tuple(inputs['pixel_values'].shape), token_shape, tuple(logits.tolist()))
