"""Pytest configuration and fixtures."""

import json
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
    # Create a simple test image with some pattern
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    
    img_path = temp_dir / "test_image.jpg"
    img.save(img_path)
    
    return img_path


@pytest.fixture
def sample_image_2(temp_dir):
    """Create a different sample test image."""
    # Create a different pattern to ensure different output
    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    
    img_path = temp_dir / "test_image_2.jpg"
    img.save(img_path)
    
    return img_path


@pytest.fixture
def haiku_vocabulary(temp_dir):
    """Create a test haiku vocabulary."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir()
    
    vocab_data = {
        "name": "test_haiku",
        "theme": "Test haiku theme",
        "output_format": "haiku",
        "words": {
            "1": ["wind", "rain", "sun", "tree", "leaf", "bird"],
            "2": ["sunset", "morning", "winter", "summer", "forest"],
            "3": ["beautiful", "whispering", "evergreen", "wandering"],
            "4": ["awakening", "meditation"],
            "5": ["metamorphosis"]
        },
        "metadata": {
            "description": "Test vocabulary",
            "version": "1.0"
        }
    }
    
    vocab_file = vocab_dir / "test_haiku.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)
    
    return vocab_dir, "test_haiku"


@pytest.fixture
def commentary_vocabulary(temp_dir):
    """Create a test commentary vocabulary."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir(exist_ok=True)
    
    vocab_data = {
        "name": "test_commentary",
        "theme": "Test commentary theme",
        "output_format": "commentary",
        "words": {
            "subjects": ["the world", "society", "people"],
            "verbs": ["evolves", "changes", "grows"],
            "descriptors": ["rapidly", "slowly", "surely"],
            "context": ["today", "now", "currently"]
        },
        "metadata": {
            "description": "Test vocabulary",
            "version": "1.0"
        }
    }
    
    vocab_file = vocab_dir / "test_commentary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)
    
    return vocab_dir, "test_commentary"


@pytest.fixture
def real_bark_image():
    """Return path to the actual bark image if it exists."""
    bark_path = Path("barks.jpg")
    if bark_path.exists():
        return bark_path
    return None

