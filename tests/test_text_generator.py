"""Tests for text generation."""

import pytest

from barkprints.text_generator import TextGenerator


def test_text_generator_initialization():
    """Test TextGenerator can be initialized."""
    generator = TextGenerator()

    assert generator.corpus_loader is not None
    assert generator.walk_generator is not None


def test_generate_with_nature_corpus(sample_image):
    """Test generating text with nature corpus."""
    generator = TextGenerator()

    text = generator.generate(str(sample_image), "nature")

    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_determinism(sample_image):
    """Test that generation is deterministic for the same image."""
    generator = TextGenerator()

    text1 = generator.generate(str(sample_image), "nature")
    text2 = generator.generate(str(sample_image), "nature")

    assert text1 == text2


def test_generate_different_images(sample_image, sample_image_2):
    """Test that different images may produce different output."""
    generator = TextGenerator()

    text1 = generator.generate(str(sample_image), "nature")
    text2 = generator.generate(str(sample_image_2), "nature")

    assert isinstance(text1, str)
    assert isinstance(text2, str)


def test_generate_different_corpora(sample_image):
    """Test generating with different corpora."""
    generator = TextGenerator()

    nature_text = generator.generate(str(sample_image), "nature")
    lit_text = generator.generate(str(sample_image), "literature")

    assert isinstance(nature_text, str)
    assert isinstance(lit_text, str)
    assert nature_text != lit_text


def test_generate_nonexistent_corpus(sample_image):
    """Test generating with non-existent corpus raises error."""
    generator = TextGenerator()

    with pytest.raises(FileNotFoundError):
        generator.generate(str(sample_image), "nonexistent")


def test_generate_with_real_bark_image(real_bark_image):
    """Test with the actual bark image if available."""
    if real_bark_image is None:
        pytest.skip("Real bark image not found")

    generator = TextGenerator()

    text = generator.generate(str(real_bark_image), "nature")

    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_with_alpha(sample_image):
    """Test generating with different alpha values."""
    gen_low = TextGenerator(alpha=0.2)
    gen_high = TextGenerator(alpha=0.8)

    text_low = gen_low.generate(str(sample_image), "nature")
    text_high = gen_high.generate(str(sample_image), "nature")

    assert isinstance(text_low, str)
    assert isinstance(text_high, str)


def test_generate_with_max_words(sample_image):
    """Test that max_words is respected."""
    generator = TextGenerator(max_words=5)

    text = generator.generate(str(sample_image), "nature")

    assert len(text.split()) <= 5
