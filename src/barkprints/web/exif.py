"""Pull GPS coordinates out of a photo's EXIF metadata.

The web picker prefers these over the device's live location: for a photo
chosen from the library, the EXIF tag records where the picture was actually
taken, whereas the browser's geolocation only knows where the phone is *now*.

Everything here is best-effort — a missing or malformed tag yields ``None``
rather than an error, since most images simply have no GPS data.
"""

from __future__ import annotations

import io
from typing import Optional, Union

from PIL import Image

# EXIF tag holding the GPS sub-IFD, and the GPS tag IDs we need within it.
_GPS_IFD = 0x8825
_GPS_LAT_REF, _GPS_LAT = 1, 2
_GPS_LON_REF, _GPS_LON = 3, 4


def _to_degrees(value) -> Optional[float]:
    """Convert an EXIF (degrees, minutes, seconds) rational triple to a float."""
    try:
        d, m, s = (float(x) for x in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return d + m / 60.0 + s / 3600.0


def extract_gps_latlon(source: Union[bytes, io.IOBase, str]) -> Optional[tuple[float, float]]:
    """Return ``(lat, lon)`` in decimal degrees from an image, or ``None``.

    ``source`` may be raw bytes, a file-like object, or a path. Any problem
    (no EXIF, no GPS IFD, missing/garbage values, out-of-range result) returns
    ``None``.
    """
    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        with Image.open(source) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps = exif.get_ifd(_GPS_IFD)
            if not gps:
                return None

            lat = gps.get(_GPS_LAT)
            lon = gps.get(_GPS_LON)
            lat_ref = gps.get(_GPS_LAT_REF)
            lon_ref = gps.get(_GPS_LON_REF)
            if not (lat and lon and lat_ref and lon_ref):
                return None

            latf = _to_degrees(lat)
            lonf = _to_degrees(lon)
            if latf is None or lonf is None:
                return None

            if str(lat_ref).strip().upper().startswith("S"):
                latf = -latf
            if str(lon_ref).strip().upper().startswith("W"):
                lonf = -lonf

            # Reject impossible coordinates and the (0, 0) null island, which is
            # almost always a zeroed-out tag rather than a real fix.
            if not (-90.0 <= latf <= 90.0 and -180.0 <= lonf <= 180.0):
                return None
            if latf == 0.0 and lonf == 0.0:
                return None
            return (latf, lonf)
    except Exception:
        return None
