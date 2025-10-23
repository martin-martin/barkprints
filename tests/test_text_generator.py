"""Tests for text generation."""

import pytest

from barkprints.text_generator import TextGenerator


def test_text_generator_initialization():
    """Test TextGenerator can be initialized."""
    generator = TextGenerator()
    
    assert generator.corpus_loader is not None
    assert generator.matcher is not None


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
    
    # They might be the same by chance with small corpora, but features should differ
    # We just verify both succeed
    assert isinstance(text1, str)
    assert isinstance(text2, str)


def test_generate_with_top_k(sample_image):
    """Test generating with top-k matches."""
    generator = TextGenerator()
    
    matches = generator.generate(str(sample_image), "nature", top_k=3)
    
    assert isinstance(matches, list)
    assert len(matches) == 3
    
    for sentence, score in matches:
        assert isinstance(sentence, str)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


def test_generate_different_corpora(sample_image):
    """Test generating with different corpora."""
    generator = TextGenerator()
    
    nature_text = generator.generate(str(sample_image), "nature")
    lit_text = generator.generate(str(sample_image), "literature")
    
    # Both should succeed
    assert isinstance(nature_text, str)
    assert isinstance(lit_text, str)
    
    # They will differ because corpora are different
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
