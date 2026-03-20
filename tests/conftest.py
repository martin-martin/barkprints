"""Pytest configuration and fixtures."""

from pathlib import Path
import tempfile

import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    img_path = temp_dir / "test_image.jpg"
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_image_2(temp_dir):
    """Create a different sample test image."""
    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    img_path = temp_dir / "test_image_2.jpg"
    img.save(img_path)

    return img_path


@pytest.fixture
def real_bark_image():
    """Return path to the actual bark image if it exists."""
    bark_path = Path("barks.jpg")
    if bark_path.exists():
        return bark_path
    return None
