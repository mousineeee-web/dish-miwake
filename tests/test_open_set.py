import hashlib
import json
import unittest

import torch

from PIL import Image

from food_catalog import (CATALOG_VERSION, CATALOG_GROUPS, DISHES, DRINKS, FRUITS,
                          ORIGINAL_DISHES, THAI_DISHES, FOOD_PROMPTS,
                          NON_FOOD_PROMPTS, OTHER_FOOD_PROMPTS, all_prompts)
from open_set import CONFIG, FoodRecognizer, decide


class OpenSetTest(unittest.TestCase):
    def scores(self):
        scores = [-10.] * len(all_prompts('English'))
        scores[0], scores[1] = 3., 1.
        scores[len(DISHES)] = 2.
        return scores

    def test_catalogue_is_60_unique_candidates_and_preserves_originals(self):
        self.assertEqual(CATALOG_VERSION, 'open60-v3')
        self.assertEqual(CONFIG['version'], CATALOG_VERSION)
        self.assertEqual(len(DISHES), 60)
        self.assertEqual(len({row[0] for row in DISHES}), 60)
        self.assertEqual(len(ORIGINAL_DISHES), 31)
        self.assertEqual(DISHES[:31], ORIGINAL_DISHES)
        self.assertEqual(hashlib.sha256(json.dumps(ORIGINAL_DISHES, ensure_ascii=False).encode()).hexdigest(),
                         '265efc7c46d0b389d8463e135e4a29d2beffb40ec761b045be4abc395b6cdaee')
        self.assertEqual([row[0] for row in DISHES[:4]], ['sushi', 'ramen', 'curry_rice', 'fried_rice'])
        self.assertEqual(len(all_prompts('English')), len(all_prompts('日本語')))
        self.assertEqual(len(all_prompts('English')), 86)

    def test_fruit_drinks_and_thai_groups_are_selectable_not_negative_foods(self):
        self.assertEqual((len(FRUITS), len(DRINKS), len(THAI_DISHES)), (12, 8, 9))
        self.assertEqual(sum(len(rows) for _, rows in CATALOG_GROUPS), len(DISHES))
        self.assertEqual(len(set(all_prompts('English'))), len(all_prompts('English')))
        self.assertNotIn('fruit', dict(OTHER_FOOD_PROMPTS))
        self.assertNotIn('drink', dict(OTHER_FOOD_PROMPTS))
        self.assertIn('a photo of fresh fruit', FOOD_PROMPTS)
        self.assertIn('a photo of a drink in a glass or a cup', FOOD_PROMPTS)
        self.assertIn('apple', [r[0] for r in FRUITS])
        self.assertIn('tea', [r[0] for r in DRINKS])
        self.assertIn('pad_thai', [r[0] for r in THAI_DISHES])

    def test_new_foods_can_be_accepted_without_bypassing_rejection(self):
        keys = [row[0] for row in DISHES]
        for key in ('apple', 'banana', 'tea', 'water', 'pad_thai', 'tom_yum'):
            with self.subTest(key=key):
                scores = self.scores()
                scores[keys.index(key)] = 5.
                self.assertEqual(decide(scores)[0], 'accepted')
                scores[-1] = 6.
                self.assertEqual(decide(scores)[0], 'non_food')

    def test_lanzhou_extension_keeps_pho_control_and_original_thresholds(self):
        self.assertIn('lanzhou_beef_noodles', [r[0] for r in DISHES])
        self.assertIn('pho', [r[0] for r in OTHER_FOOD_PROMPTS])
        self.assertEqual({key: CONFIG[key] for key in ('min_food_margin','min_dish_margin','min_other_margin','max_generic_gap','flat_image_stddev')},
                         {'min_food_margin':.5,'min_dish_margin':.35,'min_other_margin':.4,'max_generic_gap':3.,'flat_image_stddev':2.})

    def test_clear_supported_dish_is_accepted(self):
        self.assertEqual(decide(self.scores())[0], 'accepted')

    def test_non_food_dominates(self):
        scores = self.scores()
        scores[-1] = 4.
        self.assertEqual(decide(scores)[0], 'non_food')

    def test_known_unsupported_food_is_not_forced_to_dish(self):
        scores = self.scores()
        scores[len(DISHES)+len(FOOD_PROMPTS)] = 3.1
        self.assertEqual(decide(scores)[0], 'out_of_scope')

    def test_generic_food_with_poor_specific_match_is_rejected(self):
        scores = self.scores()
        scores[len(DISHES)] = 7.
        self.assertEqual(decide(scores)[0], 'out_of_scope')

    def test_near_tie_and_exact_boundary_are_uncertain(self):
        scores = self.scores()
        scores[1] = 2.9
        self.assertEqual(decide(scores)[0], 'uncertain')
        config = {**CONFIG, 'min_dish_margin': 2.}
        scores[1] = 1.
        self.assertEqual(decide(scores, config)[0], 'uncertain')

    def test_low_absolute_score_can_be_accepted_by_relative_evidence(self):
        scores = [value-10 for value in self.scores()]
        self.assertEqual(decide(scores)[0], 'accepted')

    def test_invalid_scores_are_not_silently_accepted(self):
        for scores in ([], [float('nan')]*len(all_prompts('English')),
                       [float('inf')]*len(all_prompts('English'))):
            with self.assertRaises(ValueError):
                decide(scores)

    def test_blank_image_needs_no_model_call(self):
        recognizer = FoodRecognizer(None, None)
        for color in ('white', 'black', 'red'):
            result = recognizer.predict(Image.new('RGB', (64, 64), color))
            self.assertEqual(result.decision, 'insufficient_image')
            self.assertEqual(result.predictions, ())
        with self.assertRaises(ValueError):
            recognizer.predict(Image.new('RGB', (64, 64)), 'unknown')

    def test_cached_text_features_and_real_score_formula(self):
        count = len(all_prompts('English'))

        class Processor:
            def __call__(self, text=None, images=None, **kwargs):
                return {'input_ids': torch.zeros((len(text), 64), dtype=torch.long)} if text else {
                    'pixel_values': torch.ones((1, 3, 224, 224))}

        class Model:
            calls = 0
            logit_scale = torch.tensor(0.)
            logit_bias = torch.tensor(-2.)

            def get_text_features(self, **kwargs):
                self.calls += 1
                return torch.eye(count)

            def get_image_features(self, **kwargs):
                return torch.eye(count)[0:1]

        model = Model()
        recognizer = FoodRecognizer(Processor(), model)
        photo = Image.linear_gradient('L').resize((64, 64)).convert('RGB')
        first, second = recognizer.predict(photo), recognizer.predict(photo)
        self.assertEqual(model.calls, 1)
        self.assertEqual(first.raw_logits, second.raw_logits)
        self.assertEqual(first.raw_logits[0], -1.)
        self.assertEqual(first.raw_logits[1], -2.)
        self.assertEqual(first.input_ids_shape, (count, 64))
        self.assertEqual(first.decision, 'accepted')


if __name__ == '__main__':
    unittest.main()
