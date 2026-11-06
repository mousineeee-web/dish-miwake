import io
import unittest
from unittest.mock import patch

from PIL import Image

from image_input import (MAX_FILE_SIZE, MAX_IMAGE_PIXELS, MAX_OUTPUT_EDGE,
                         SUPPORTED_FORMATS, UPLOAD_EXTENSIONS, ImageUploadError,
                         _check_dimensions, decode_uploaded_image, open_uploaded_image)


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

    def test_real_bytes_for_all_supported_raster_decoders(self):
        for fmt in SUPPORTED_FORMATS:
            with self.subTest(format=fmt):
                result = decode_uploaded_image(encoded_image(fmt))
                self.assertEqual(result.detected_format, fmt)
                self.assertEqual(result.original_size, (16, 12))
                self.assertEqual(result.image.mode, 'RGB')
                self.assertEqual(result.image.info, {})
                self.assertFalse(result.resized)

    def test_only_approved_raster_decoders_are_tried(self):
        with patch('image_input.Image.open', side_effect=OSError) as decoder:
            with self.assertRaises(ImageUploadError):
                open_uploaded_image(b'%!PS-Adobe-3.0 EPSF-3.0')
        self.assertEqual(decoder.call_args.kwargs['formats'], SUPPORTED_FORMATS)
        self.assertNotIn('EPS', SUPPORTED_FORMATS)

    def test_svg_pdf_and_html_are_not_images_even_if_renamed(self):
        for content in (b'<svg xmlns="http://www.w3.org/2000/svg"/>',
                        b'%PDF-1.7', b'<html><script>alert(1)</script></html>'):
            with self.subTest(content=content):
                with self.assertRaises(ImageUploadError):
                    open_uploaded_image(content)

    def test_large_pixel_count_is_rejected_before_decode(self):
        with patch("image_input.Image.open") as decoder:
            image = decoder.return_value.__enter__.return_value
            image.format, image.width, image.height = "PNG", MAX_IMAGE_PIXELS + 1, 1
            with self.assertRaises(ValueError):
                open_uploaded_image(b"image header")
            image.load.assert_not_called()

    def test_aliases_come_from_registered_raster_plugins(self):
        for alias in ('jpeg', 'jfif', 'heic', 'heif', 'hif', 'webp', 'avif', 'tif', 'jp2'):
            self.assertIn(alias, UPLOAD_EXTENSIONS)

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

    def test_common_12_24_and_48_megapixel_dimensions_are_allowed(self):
        from types import SimpleNamespace
        for width, height in ((4032, 3024), (5712, 4284), (8064, 6048)):
            _check_dimensions(SimpleNamespace(width=width, height=height))
        with self.assertRaisesRegex(ImageUploadError, '5000万画素'):
            _check_dimensions(SimpleNamespace(width=MAX_IMAGE_PIXELS + 1, height=1))
        with self.assertRaises(ImageUploadError):
            _check_dimensions(SimpleNamespace(width=0, height=12))

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

    def test_animation_and_multipage_only_use_first_image(self):
        for fmt in ('GIF', 'WEBP', 'PNG', 'TIFF', 'AVIF'):
            with self.subTest(format=fmt):
                output = io.BytesIO()
                Image.new('RGB', (16, 12), 'red').save(
                    output, format=fmt, save_all=True,
                    append_images=[Image.new('RGB', (16, 12), 'blue')], duration=100)
                image = open_uploaded_image(output.getvalue())
                red, green, blue = image.getpixel((8, 6))
                self.assertGreater(red, blue + 150)

    def test_transparent_and_palette_images_have_white_background(self):
        for fmt in ('PNG', 'WEBP', 'AVIF', 'TIFF', 'GIF'):
            with self.subTest(format=fmt):
                output = io.BytesIO()
                source = Image.new('RGBA', (16, 12), (255, 0, 0, 0))
                source.save(output, format=fmt)
                self.assertEqual(open_uploaded_image(output.getvalue()).getpixel((4, 4)),
                                 (255, 255, 255))

    def test_partial_alpha_is_composited_not_discarded(self):
        output = io.BytesIO()
        Image.new('RGBA', (16, 12), (255, 0, 0, 128)).save(output, format='PNG')
        self.assertEqual(open_uploaded_image(output.getvalue()).getpixel((4, 4)),
                         (255, 127, 127))

    def test_cmyk_jpeg_and_16bit_grayscale_tiff(self):
        for fmt, mode, color, expected in (
            ('JPEG', 'CMYK', (0, 255, 255, 0), (255, 0, 0)),
            ('TIFF', 'I;16', 32768, (127, 127, 127)),
        ):
            with self.subTest(format=fmt, mode=mode):
                output = io.BytesIO()
                Image.new(mode, (16, 12), color).save(output, format=fmt)
                self.assertEqual(open_uploaded_image(output.getvalue()).getpixel((4, 4)), expected)

    def test_large_photo_is_resized_with_aspect_ratio_and_rotation(self):
        for fmt in ('JPEG', 'PNG', 'HEIF', 'TIFF'):
            with self.subTest(format=fmt):
                output = io.BytesIO()
                source = Image.new('RGB', (3072, 1536), 'red')
                exif = Image.Exif()
                exif[274] = 6
                source.save(output, format=fmt, exif=exif.tobytes())
                result = decode_uploaded_image(output.getvalue())
                self.assertTrue(result.resized)
                self.assertEqual(max(result.image.size), MAX_OUTPUT_EDGE)
                self.assertEqual(result.image.size, (1024, 2048))
                self.assertEqual(result.image.info, {})

    def test_gps_exif_and_xmp_are_not_forwarded(self):
        output = io.BytesIO()
        exif = Image.Exif()
        exif[270] = 'private description'
        exif[34853] = {1: 'N', 2: (35.0, 0.0, 0.0)}
        Image.new('RGB', (16, 12)).save(output, format='JPEG', exif=exif.tobytes(),
                                        xmp=b'private location')
        result = open_uploaded_image(output.getvalue())
        self.assertEqual(result.info, {})
        self.assertFalse(result.getexif())

    def test_truncated_jpeg_is_not_silently_accepted(self):
        with self.assertRaises(ImageUploadError):
            open_uploaded_image(encoded_image('JPEG')[:-40])


if __name__ == "__main__":
    unittest.main()
