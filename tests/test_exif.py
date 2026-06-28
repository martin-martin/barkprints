"""Tests for GPS extraction from photo EXIF metadata."""

import io

import pytest
from PIL import Image

from barkprints.web.exif import extract_gps_latlon


def _jpeg_with_gps(lat_ref, lat_dms, lon_ref, lon_dms) -> bytes:
    img = Image.new("RGB", (8, 8), (120, 90, 60))
    exif = Image.Exif()
    exif[0x8825] = {1: lat_ref, 2: lat_dms, 3: lon_ref, 4: lon_dms}
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_extracts_northeast_coordinates():
    data = _jpeg_with_gps("N", (47.0, 4.0, 14.52), "E", (15.0, 26.0, 22.2))
    result = extract_gps_latlon(data)
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(47.0707, abs=1e-4)
    assert lon == pytest.approx(15.4395, abs=1e-4)


def test_south_and_west_are_negative():
    data = _jpeg_with_gps("S", (33.0, 52.0, 4.0), "W", (151.0, 12.0, 36.0))
    result = extract_gps_latlon(data)
    assert result is not None
    lat, lon = result
    assert lat < 0 and lon < 0
    assert lat == pytest.approx(-33.8678, abs=1e-4)
    assert lon == pytest.approx(-151.21, abs=1e-4)


def test_image_without_exif_returns_none():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    assert extract_gps_latlon(buf.getvalue()) is None


def test_null_island_is_rejected():
    # All-zero GPS tags are a zeroed-out field, not a real fix in the ocean.
    data = _jpeg_with_gps("N", (0.0, 0.0, 0.0), "E", (0.0, 0.0, 0.0))
    assert extract_gps_latlon(data) is None


def test_garbage_input_returns_none():
    assert extract_gps_latlon(b"not an image") is None


def test_accepts_file_like_object():
    data = _jpeg_with_gps("N", (10.0, 0.0, 0.0), "E", (20.0, 0.0, 0.0))
    result = extract_gps_latlon(io.BytesIO(data))
    assert result == pytest.approx((10.0, 20.0))
