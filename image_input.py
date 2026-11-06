"""Content-sniffed raster uploads, normalized for preview and inference in memory."""
from dataclasses import dataclass
from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

# Keep libheif security limits. Decode the primary photo, never depth/aux images.
register_heif_opener(thumbnails=False, depth_images=False, aux_images=False,
                     decode_threads=1)  # The Pillow plugin decodes HDR to 8-bit.

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_OUTPUT_EDGE = 2048
# Explicit raster decoders only: no EPS/Ghostscript, PDF, SVG, camera RAW or video.
SUPPORTED_FORMATS = ("JPEG", "PNG", "HEIF", "WEBP", "AVIF", "GIF", "TIFF",
                     "BMP", "DIB", "JPEG2000")
SUPPORTED_FORMAT_LABEL = "JPEG / PNG / HEIC・HEIF / WebP / AVIF / GIF / TIFF / BMP / JPEG 2000"
# Informational aliases, NOT an upload/security filter. The bytes are decisive.
UPLOAD_EXTENSIONS = tuple(sorted(
    extension.lstrip(".") for extension, fmt in Image.registered_extensions().items()
    if fmt in SUPPORTED_FORMATS
))
_DECODE_LOCK = Lock()


class ImageUploadError(ValueError):
    """Safe user-facing validation message; never expose decoder internals."""


@dataclass(frozen=True)
class DecodedUpload:
    image: Image.Image
    detected_format: str
    original_size: tuple[int, int]
    resized: bool


def _check_dimensions(image):
    if image.width <= 0 or image.height <= 0:
        raise ImageUploadError("画像の大きさが不正です。別の写真を選んでください。")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageUploadError("画像が5000万画素を超えています。写真を縮小してから再度選んでください。")


def _to_rgb(image):
    # 16-bit grayscale must be scaled, not clipped to white by convert('RGB').
    if image.mode.startswith("I;16"):
        image = image.convert("F").point(lambda value: value / 257).convert("L")
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")  # Includes CMYK, grayscale and palette images.


def decode_uploaded_image(file_bytes: bytes) -> DecodedUpload:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ImageUploadError("画像が20 MiBを超えています。小さい画像を選んでください。")
    try:
        # Bound concurrent full-resolution decodes in one server process.
        with _DECODE_LOCK, Image.open(BytesIO(file_bytes), formats=SUPPORTED_FORMATS) as image:
            _check_dimensions(image)  # Inspect the header before allocating pixels.
            source_format, original_size = image.format, image.size
            # Non-HEIF plugins start on frame/page zero. Do not enumerate n_frames:
            # TIFF can traverse every IFD. HEIF already selects its primary photo.
            resized = max(original_size) > MAX_OUTPUT_EDGE
            if resized:
                # Uses decoder draft reduction for JPEG where available. Other
                # codecs may still need to decode the full, pixel-limited source.
                image.thumbnail((MAX_OUTPUT_EDGE, MAX_OUTPUT_EDGE), Image.Resampling.LANCZOS)
            image.load()
            _check_dimensions(image)
            normalized = _to_rgb(ImageOps.exif_transpose(image))
            normalized.info.clear()  # Never forward EXIF/GPS/XMP to the preview.
            return DecodedUpload(normalized, source_format, original_size, resized)
    except ImageUploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError, EOFError,
            SyntaxError, Image.DecompressionBombError):
        raise ImageUploadError(
            "画像を読み込めませんでした。対応する写真を選んでください。"
            "破損・未対応の形式（RAW・SVG・PDF・動画など）はJPEG/PNGへ書き出してください。"
            "拡張子の変更だけでは変換できません。"
        ) from None


def open_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Backward-compatible image-only entry point."""
    return decode_uploaded_image(file_bytes).image
