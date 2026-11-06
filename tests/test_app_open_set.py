import io
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from classifier import Prediction
from food_catalog import DISHES
from image_input import MAX_FILE_SIZE
from open_set import FoodResult

APP = Path(__file__).resolve().parents[1] / 'app.py'


class Upload:
    name = 'sample.jpg'

    def __init__(self, content, size=None):
        self.content = content
        self.size = len(content) if size is None else size

    def getvalue(self):
        return self.content


class AppOpenSetTest(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        output = io.BytesIO()
        Image.linear_gradient('L').resize((64, 64)).convert('RGB').save(output, format='JPEG')
        self.upload = Upload(output.getvalue())

    def run_result(self, decision):
        result = FoodResult(decision, '判断を保留する説明' if decision != 'accepted' else '参考候補です',
                            (Prediction('pizza', 'ピザ', 'a photo of pizza', .8),), {}, .1)
        with patch('streamlit.file_uploader', return_value=self.upload), \
             patch('classifier.load_runtime', return_value=(None, None)), \
             patch('open_set.FoodRecognizer.predict', return_value=result):
            app = AppTest.from_file(str(APP)).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_accepted_dish_is_visible(self):
        app = self.run_result('accepted')
        self.assertTrue(any('食べ物・飲み物の候補：「ピザ」' in node.value for node in app.success))

    def test_rejected_images_never_show_a_successful_dish_name(self):
        for decision in ('non_food', 'out_of_scope', 'uncertain', 'insufficient_image'):
            st.cache_resource.clear()
            app = self.run_result(decision)
            self.assertTrue(any('判定を保留しました' in node.value for node in app.warning))
            self.assertFalse(any('食べ物・飲み物の候補' in node.value for node in app.success))

    def test_initial_screen_has_current_dish_scope(self):
        with patch('streamlit.file_uploader', return_value=None):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(f'{len(DISHES)}種類' in node.value for node in app.markdown))
        self.assertTrue(any('追加学習する仕組みではありません' in node.value for node in app.caption))

    def test_file_picker_and_preview_accept_real_heic_bytes(self):
        output = io.BytesIO()
        from image_input import open_uploaded_image
        Image.linear_gradient('L').resize((64, 48)).convert('RGB').save(output, format='HEIF')
        self.assertEqual(open_uploaded_image(output.getvalue()).size, (64, 48))
        upload = Upload(output.getvalue())
        upload.name = 'IMG_0001.HEIC'
        with patch('streamlit.file_uploader', return_value=upload) as picker, \
             patch('classifier.load_runtime') as load:
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('読み込みました' in node.value for node in app.success))
        self.assertIsNone(picker.call_args.kwargs['type'])
        self.assertEqual(picker.call_args.kwargs['max_upload_size'], 20)
        load.assert_not_called()

    def test_corrupt_upload_is_rejected_before_runtime(self):
        with patch('streamlit.file_uploader', return_value=Upload(b'not an image')), \
             patch('classifier.load_runtime') as load:
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('読み込めません' in node.value for node in app.error))
        load.assert_not_called()

    def test_oversized_upload_is_rejected_before_runtime(self):
        with patch('streamlit.file_uploader', return_value=Upload(b'', MAX_FILE_SIZE+1)), \
             patch('classifier.load_runtime') as load:
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('20 MiB' in node.value for node in app.error))
        load.assert_not_called()

    def test_real_content_not_filename_or_mime_controls_decoding(self):
        for name in ('IMG_1234.jpeg', 'IMG_1234.JFIF', 'IMG_1234.HEIC',
                     'download', '写真.bin'):
            with self.subTest(name=name):
                upload = Upload(self.upload.content)
                upload.name = name
                upload.type = 'application/octet-stream'
                with patch('streamlit.file_uploader', return_value=upload), \
                     patch('classifier.load_runtime') as load:
                    app = AppTest.from_file(str(APP)).run(timeout=20)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('読み込みました' in node.value for node in app.success))
                self.assertTrue(any('検出形式：JPEG' in node.value for node in app.caption))
                load.assert_not_called()

    def test_all_common_formats_preview_without_loading_the_model(self):
        for fmt in ('PNG', 'WEBP', 'AVIF', 'GIF', 'TIFF', 'BMP', 'JPEG2000'):
            with self.subTest(format=fmt):
                output = io.BytesIO()
                Image.new('RGB', (64, 48), 'red').save(output, format=fmt)
                with patch('streamlit.file_uploader', return_value=Upload(output.getvalue())), \
                     patch('classifier.load_runtime') as load:
                    app = AppTest.from_file(str(APP)).run(timeout=20)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any('検出形式：' + fmt in node.value for node in app.caption))
                load.assert_not_called()

    def test_frame_policy_and_unsupported_formats_are_explained(self):
        with patch('streamlit.file_uploader', return_value=None):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertTrue(any('先頭の1枚' in node.value and 'HEIFは主画像' in node.value
                            for node in app.caption))
        self.assertTrue(any('RAW・SVG・PDF・動画' in node.value for node in app.caption))

    def test_water_suggestion_has_visual_only_warning(self):
        result = FoodResult('accepted', '見た目の候補です',
                            (Prediction('water', '水（見た目の候補）', 'a photo of water', .8),),
                            {}, .1)
        with patch('streamlit.file_uploader', return_value=self.upload), \
             patch('classifier.load_runtime', return_value=(None, None)), \
             patch('open_set.FoodRecognizer.predict', return_value=result):
            app = AppTest.from_file(str(APP)).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('飲用の安全性を確認できません' in node.value for node in app.warning))

    def test_internal_error_details_are_not_shown_to_visitors(self):
        private_error = 'Internal path /private/server/cache: do not expose'
        with patch('streamlit.file_uploader', return_value=self.upload), \
             patch('classifier.load_runtime', side_effect=RuntimeError(private_error)):
            app = AppTest.from_file(str(APP)).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('推定を実行できませんでした' in node.value for node in app.error))
        self.assertFalse(any(private_error in node.value for node in app.code))

    def test_privacy_notice_is_visible_before_upload(self):
        with patch('streamlit.file_uploader', return_value=None):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertTrue(any('実行サーバーへ送信' in node.value for node in app.caption))


if __name__ == '__main__':
    unittest.main()
