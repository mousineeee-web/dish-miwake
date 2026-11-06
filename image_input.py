"""Validate public uploads before decoding large images or loading the model."""
from io import BytesIO

from PIL import Image

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


def open_uploaded_image(file_bytes: bytes) -> Image.Image:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("Image exceeds 10 MiB")
    with Image.open(BytesIO(file_bytes)) as image:
        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Only JPEG and PNG images are supported")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds 12 megapixels")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated images are not supported")
        image.load()
        return image.convert("RGB")
