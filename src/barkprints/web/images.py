"""Server-side image handling: HEIC normalization and gallery thumbnails.

Browsers can't render HEIC/HEIF in an ``<img>`` tag, so uploads in those
formats are re-encoded to JPEG once at save time. Thumbnails are small JPEGs
generated lazily under ``<data_dir>/thumbs`` so the gallery list and map
popups don't ship multi-megabyte originals to a phone.

Importing this module registers the pillow-heif opener (when installed) so
``PIL.Image.open`` — used by generation, EXIF extraction, and thumbnailing —
can decode HEIC anywhere in the process.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:  # pragma: no cover - pillow-heif ships with the web extra
    HEIF_SUPPORTED = False

# Big enough to stay sharp as the gallery card image on a high-DPI phone,
# ~two orders of magnitude smaller than a full camera photo.
THUMB_MAX_DIM = 800
THUMB_QUALITY = 82
JPEG_QUALITY = 90


def to_jpeg_bytes(data: bytes) -> bytes:
    """Re-encode image bytes as JPEG.

    Orientation is baked in via ``exif_transpose`` (which also drops the
    orientation tag from the copied EXIF, so viewers won't rotate twice);
    the rest of the EXIF — including GPS — is carried over. Raises on
    undecodable input.
    """
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        exif = img.getexif().tobytes()
        out = io.BytesIO()
        img.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY, exif=exif)
        return out.getvalue()


def make_thumbnail(src: Path, dest: Path) -> bool:
    """Write a small JPEG thumbnail of ``src`` to ``dest``.

    Returns False when the source can't be decoded (missing codec, corrupt
    file) so the caller can fall back to serving the original.
    """
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.Resampling.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(dest, "JPEG", quality=THUMB_QUALITY)
        return True
    except Exception:
        return False
