"""Integration tests for the entire system."""

import pytest

from barkprints.text_generator import TextGenerator


def test_end_to_end_nature(sample_image):
    """Test complete flow: image -> nature corpus -> generative text."""
    generator = TextGenerator()

    text = generator.generate(str(sample_image), "nature")

    assert isinstance(text, str)
    assert len(text) > 5


def test_end_to_end_literature(sample_image):
    """Test complete flow: image -> literature corpus -> generative text."""
    generator = TextGenerator()

    text = generator.generate(str(sample_image), "literature")

    assert isinstance(text, str)
    assert len(text) > 5


def test_multiple_images_maintain_determinism(sample_image, sample_image_2):
    """Test that multiple images each produce deterministic output."""
    generator = TextGenerator()

    text1a = generator.generate(str(sample_image), "nature")
    text1b = generator.generate(str(sample_image), "nature")
    assert text1a == text1b

    text2a = generator.generate(str(sample_image_2), "nature")
    text2b = generator.generate(str(sample_image_2), "nature")
    assert text2a == text2b


def test_corpus_switching(sample_image):
    """Test switching between corpora."""
    generator = TextGenerator()

    nature1 = generator.generate(str(sample_image), "nature")
    lit1 = generator.generate(str(sample_image), "literature")
    nature2 = generator.generate(str(sample_image), "nature")

    assert nature1 == nature2
    assert nature1 != lit1


def test_real_world_usage_pattern():
    """Test a realistic usage pattern."""
    bark_path = "barks.jpg"

    try:
        generator = TextGenerator()

        nature_voice = generator.generate(bark_path, "nature")
        lit_voice = generator.generate(bark_path, "literature")

        assert len(nature_voice) > 0
        assert len(lit_voice) > 0

        # Same image+corpus should always produce same output
        nature_voice2 = generator.generate(bark_path, "nature")
        assert nature_voice == nature_voice2

    except FileNotFoundError:
        pytest.skip("Real bark image not available")
