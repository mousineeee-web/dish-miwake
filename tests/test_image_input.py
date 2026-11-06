import io
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from image_input import MAX_FILE_SIZE, MAX_IMAGE_PIXELS, open_uploaded_image


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
        with self.assertRaises(UnidentifiedImageError):
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

    def test_animation_is_rejected_before_decode(self):
        with patch("image_input.Image.open") as decoder:
            image = decoder.return_value.__enter__.return_value
            image.format, image.width, image.height, image.n_frames = "PNG", 16, 12, 2
            with self.assertRaises(ValueError):
                open_uploaded_image(b"image header")
            image.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
