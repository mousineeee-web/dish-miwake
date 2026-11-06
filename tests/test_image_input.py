import io
import unittest
from unittest.mock import patch

from PIL import Image

from image_input import (MAX_FILE_SIZE, MAX_IMAGE_PIXELS, UPLOAD_EXTENSIONS,
                         ImageUploadError, _check_dimensions, open_uploaded_image)


def encoded_image(fmt="PNG", mode="RGB"):
    output = io.BytesIO()
    Image.new(mode, (16, 12)).save(output, format=fmt)
    return output.getvalue()


class ImageInputTest(unittest.TestCase):
    def test_jpeg_png_and_rgb_conversion(self):
        for fmt, mode in (("JPEG", "RGB"), ("PNG", "RGBA"), ("PNG", "L")):
            image = open_uploaded_image(encoded_image(fmt, mode))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (16, 12))
            image.load()

    def test_rejects_file_too_large_before_decoding(self):
        with patch("image_input.Image.open") as decoder:
            with self.assertRaises(ValueError):
                open_uploaded_image(b"x" * (MAX_FILE_SIZE + 1))
        decoder.assert_not_called()

    def test_rejects_non_images(self):
        with self.assertRaisesRegex(ImageUploadError, '読み込めません'):
            open_uploaded_image(b"not an image")

    def test_gif_cannot_bypass_extension_filter(self):
        with self.assertRaises(ValueError):
            open_uploaded_image(encoded_image("GIF"))

    def test_large_pixel_count_is_rejected_before_decode(self):
        with patch("image_input.Image.open") as decoder:
            image = decoder.return_value.__enter__.return_value
            image.format, image.width, image.height = "PNG", MAX_IMAGE_PIXELS + 1, 1
            with self.assertRaises(ValueError):
                open_uploaded_image(b"image header")
            image.load.assert_not_called()

    def test_heic_and_heif_are_selectable_extensions(self):
        self.assertIn('heic', UPLOAD_EXTENSIONS)
        self.assertIn('heif', UPLOAD_EXTENSIONS)

    def test_real_heif_bytes_decode_as_rgb(self):
        image = open_uploaded_image(encoded_image('HEIF'))
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, (16, 12))
        self.assertEqual(image.info, {})

    def test_heif_uses_primary_photo_not_first_frame(self):
        output = io.BytesIO()
        thumbnail = Image.new('RGB', (16, 12), 'red')
        primary = Image.new('RGB', (32, 24), 'lime')
        thumbnail.save(output, format='HEIF', save_all=True,
                       append_images=[primary], primary_index=1, quality=95)
        image = open_uploaded_image(output.getvalue())
        self.assertEqual(image.size, (32, 24))
        red, green, blue = image.getpixel((16, 12))
        self.assertGreater(green, red + 100)
        self.assertGreater(green, blue + 100)

    def test_phone_exif_orientation_is_applied_once_for_jpeg_and_heif(self):
        for fmt in ('JPEG', 'HEIF'):
            with self.subTest(format=fmt):
                output = io.BytesIO()
                image = Image.new('RGB', (16, 12), 'red')
                exif = Image.Exif()
                exif[274] = 6
                image.save(output, format=fmt, exif=exif.tobytes(), quality=95)
                result = open_uploaded_image(output.getvalue())
                self.assertEqual(result.size, (12, 16))
                self.assertFalse(result.getexif())

    def test_common_12_and_24_megapixel_dimensions_are_allowed(self):
        from types import SimpleNamespace
        for width, height in ((4032, 3024), (5712, 4284)):
            _check_dimensions(SimpleNamespace(width=width, height=height))
        with self.assertRaisesRegex(ImageUploadError, '2500万画素'):
            _check_dimensions(SimpleNamespace(width=8064, height=6048))

    def test_ordinary_jpeg_pixel_values_do_not_change(self):
        data = encoded_image('JPEG')
        with Image.open(io.BytesIO(data)) as old:
            expected = old.convert('RGB').tobytes()
        self.assertEqual(open_uploaded_image(data).tobytes(), expected)

    def test_decoder_error_is_replaced_with_safe_message(self):
        with patch('image_input.Image.open', side_effect=RuntimeError('private decoder path')):
            with self.assertRaises(ImageUploadError) as result:
                open_uploaded_image(b'corrupt heic')
        self.assertNotIn('private decoder path', str(result.exception))

    def test_animation_is_rejected_before_decode(self):
        with patch("image_input.Image.open") as decoder:
            image = decoder.return_value.__enter__.return_value
            image.format, image.width, image.height, image.n_frames = "PNG", 16, 12, 2
            with self.assertRaises(ValueError):
                open_uploaded_image(b"image header")
            image.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
