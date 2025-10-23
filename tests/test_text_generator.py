"""Tests for text generation."""

import pytest

from barkprints.text_generator import TextGenerator


def test_text_generator_initialization():
    """Test TextGenerator can be initialized."""
    generator = TextGenerator()
    
    assert generator.vocabulary_loader is not None
    assert generator.format_loader is not None


def test_generate_with_custom_vocabulary(sample_image, haiku_vocabulary):
    """Test generating text with a custom vocabulary."""
    vocab_dir, vocab_name = haiku_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    text = generator.generate(str(sample_image), vocab_name)
    
    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_determinism(sample_image, haiku_vocabulary):
    """Test that generation is deterministic for the same image."""
    vocab_dir, vocab_name = haiku_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    text1 = generator.generate(str(sample_image), vocab_name)
    text2 = generator.generate(str(sample_image), vocab_name)
    
    assert text1 == text2


def test_generate_different_images_different_output(
    sample_image, sample_image_2, haiku_vocabulary
):
    """Test that different images produce different output."""
    vocab_dir, vocab_name = haiku_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    text1 = generator.generate(str(sample_image), vocab_name)
    text2 = generator.generate(str(sample_image_2), vocab_name)
    
    # Different images should (very likely) produce different text
    assert text1 != text2


def test_generate_with_format_override(sample_image, haiku_vocabulary):
    """Test generating with format override."""
    vocab_dir, vocab_name = haiku_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    # Override to use sentence format instead of haiku
    text = generator.generate(str(sample_image), vocab_name, "sentence")
    
    assert isinstance(text, str)
    # Should be single line (sentence format)
    assert '\n' not in text


def test_generate_with_commentary_vocabulary(sample_image, commentary_vocabulary):
    """Test generating with commentary vocabulary."""
    vocab_dir, vocab_name = commentary_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    text = generator.generate(str(sample_image), vocab_name)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.endswith('.')


def test_generate_with_nonexistent_vocabulary(sample_image, temp_dir):
    """Test generating with non-existent vocabulary raises error."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir()
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    with pytest.raises(FileNotFoundError):
        generator.generate(str(sample_image), "nonexistent")


def test_generate_with_nonexistent_format(sample_image, haiku_vocabulary):
    """Test generating with non-existent format raises error."""
    vocab_dir, vocab_name = haiku_vocabulary
    
    generator = TextGenerator()
    generator.vocabulary_loader.vocabularies_dir = vocab_dir
    
    with pytest.raises(KeyError):
        generator.generate(str(sample_image), vocab_name, "nonexistent")


def test_generate_with_real_bark_image(real_bark_image):
    """Test with the actual bark image if available."""
    if real_bark_image is None:
        pytest.skip("Real bark image not found")
    
    generator = TextGenerator()
    
    # Test with nature vocabulary
    text = generator.generate(str(real_bark_image), "nature")
    
    assert isinstance(text, str)
    assert len(text) > 0
    
    # Should be a haiku (3 lines)
    lines = text.split('\n')
    assert len(lines) == 3

