"""Integration tests for the entire system."""

import pytest

from barkprints.text_generator import TextGenerator


def test_end_to_end_nature(sample_image):
    """Test complete flow: image -> nature corpus -> text."""
    generator = TextGenerator()
    
    text = generator.generate(str(sample_image), "nature")
    
    # Verify we get actual text
    assert isinstance(text, str)
    assert len(text) > 10  # Should be a real sentence
    assert text.endswith('.')  # Should be properly punctuated


def test_end_to_end_literature(sample_image):
    """Test complete flow: image -> literature corpus -> text."""
    generator = TextGenerator()
    
    text = generator.generate(str(sample_image), "literature")
    
    assert isinstance(text, str)
    assert len(text) > 10


def test_multiple_images_maintain_determinism(sample_image, sample_image_2):
    """Test that multiple images each produce deterministic output."""
    generator = TextGenerator()
    
    # Image 1
    text1a = generator.generate(str(sample_image), "nature")
    text1b = generator.generate(str(sample_image), "nature")
    assert text1a == text1b
    
    # Image 2
    text2a = generator.generate(str(sample_image_2), "nature")
    text2b = generator.generate(str(sample_image_2), "nature")
    assert text2a == text2b


def test_corpus_switching(sample_image):
    """Test switching between corpora."""
    generator = TextGenerator()
    
    # Generate with different corpora
    nature1 = generator.generate(str(sample_image), "nature")
    lit1 = generator.generate(str(sample_image), "literature")
    nature2 = generator.generate(str(sample_image), "nature")
    
    # Should be consistent per corpus
    assert nature1 == nature2
    
    # Should differ between corpora
    assert nature1 != lit1


def test_real_world_usage_pattern():
    """Test a realistic usage pattern."""
    bark_path = "barks.jpg"
    
    try:
        generator = TextGenerator()
        
        # Get different "voices" for the same bark
        nature_voice = generator.generate(bark_path, "nature")
        lit_voice = generator.generate(bark_path, "literature")
        
        # All should succeed and return text
        assert len(nature_voice) > 0
        assert len(lit_voice) > 0
        
        # Get top matches
        top_matches = generator.generate(bark_path, "nature", top_k=3)
        assert len(top_matches) == 3
        assert all(isinstance(s, str) and isinstance(sc, float) for s, sc in top_matches)
        
        # Same image+corpus should always produce same output
        nature_voice2 = generator.generate(bark_path, "nature")
        assert nature_voice == nature_voice2
        
    except FileNotFoundError:
        pytest.skip("Real bark image not available")
