"""Decode phone photos safely; preserve the HEIF primary image and orientation."""
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

# Decode the full primary photo, not a thumbnail, depth map or auxiliary image.
# libheif's own security limits remain enabled.
register_heif_opener(thumbnails=False, depth_images=False, aux_images=False,
                     decode_threads=1)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
UPLOAD_EXTENSIONS = ("jpg", "jpeg", "png", "heic", "heif")


class ImageUploadError(ValueError):
    """A safe, user-facing validation message; never expose decoder internals."""


def _check_dimensions(image):
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageUploadError("画像が2500万画素を超えています。写真を縮小してから再度選んでください。")


def open_uploaded_image(file_bytes: bytes) -> Image.Image:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ImageUploadError("画像が10 MBを超えています。小さい画像を選んでください。")
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            if image.format not in ("JPEG", "PNG", "HEIF"):
                raise ImageUploadError("JPG・PNG・HEIC・HEIF形式の写真を選んでください。")
            _check_dimensions(image)
            if image.format != "HEIF" and getattr(image, "n_frames", 1) != 1:
                raise ImageUploadError("アニメーションではなく静止画を選んでください。")
            # Pillow's HEIF plugin starts on the primary image. Do not seek(0):
            # a phone container can include additional images before its main photo.
            image.load()
            _check_dimensions(image)
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.info.clear()  # Do not forward EXIF/GPS or XMP to the preview.
            return normalized
    except ImageUploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError, EOFError,
            SyntaxError, Image.DecompressionBombError):
        raise ImageUploadError("画像を読み込めませんでした。破損していないJPG・PNG・HEIC・HEIF形式の写真を選んでください。") from None
