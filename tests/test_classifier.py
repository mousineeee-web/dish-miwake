import unittest

import torch
from PIL import Image

from classifier import get_prompts, predict


class FakeProcessor:
    def __call__(self, **_kwargs):
        return {
            "pixel_values": torch.zeros((1, 3, 224, 224)),
            "input_ids": torch.zeros((4, 64), dtype=torch.long),
        }


class FakeOutput:
    logits_per_image = torch.tensor([[0.0, 3.0, -2.0, 1.0]])


class FakeModel:
    def __call__(self, **_kwargs):
        return FakeOutput()


class ClassifierTest(unittest.TestCase):
    def test_prompt_order_is_stable(self):
        self.assertEqual(
            get_prompts("English"),
            (
                "a photo of sushi",
                "a photo of ramen",
                "a photo of Japanese curry rice",
                "a photo of fried rice",
            ),
        )

    def test_predictions_are_sorted_by_score(self):
        image = Image.new("RGB", (32, 24), "white")
        result = predict(image, FakeProcessor(), FakeModel(), "English")

        self.assertEqual(
            [prediction.key for prediction in result.predictions],
            ["ramen", "fried_rice", "sushi", "curry_rice"],
        )
        self.assertEqual(result.pixel_values_shape, (1, 3, 224, 224))
        self.assertEqual(result.input_ids_shape, (4, 64))

    def test_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            get_prompts("French")


if __name__ == "__main__":
    unittest.main()
