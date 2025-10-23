"""Integration tests for the entire system."""

import pytest

from barkprints.text_generator import TextGenerator


def test_end_to_end_nature_haiku(sample_image, temp_dir):
    """Test complete flow: image -> nature vocabulary -> haiku."""
    # Use the built-in nature vocabulary
    generator = TextGenerator()
    
    text = generator.generate(str(sample_image), "nature")
    
    # Verify it's a haiku (3 lines)
    lines = text.split('\n')
    assert len(lines) == 3
    
    # Each line should have content
    for line in lines:
        assert len(line.strip()) > 0
        # Lines should be reasonable length
        assert len(line.split()) >= 1


def test_end_to_end_news_commentary(sample_image):
    """Test complete flow: image -> news vocabulary -> commentary."""
    generator = TextGenerator()
    
    text = generator.generate(str(sample_image), "news")
    
    # Should be a single sentence
    assert text.endswith('.')
    assert len(text.split()) >= 3  # At least a few words


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
    
    # But images should produce different output
    assert text1a != text2a


def test_vocabulary_and_format_compatibility():
    """Test that format overrides work correctly."""
    generator = TextGenerator()
    
    # This should work (haiku vocab, sentence format)
    text = generator.generate("barks.jpg", "nature", "sentence")
    assert '\n' not in text  # Sentence format = single line
    
    # This should fail (news vocab, haiku format)
    with pytest.raises(ValueError):
        generator.generate("barks.jpg", "news", "haiku")


def test_real_world_usage_pattern():
    """Test a realistic usage pattern."""
    bark_path = "barks.jpg"
    
    try:
        generator = TextGenerator()
        
        # Generate different representations of the same bark
        haiku = generator.generate(bark_path, "nature", "haiku")
        sentence = generator.generate(bark_path, "nature", "sentence")
        news = generator.generate(bark_path, "news", "commentary")
        
        # All should succeed and return text
        assert len(haiku) > 0
        assert len(sentence) > 0
        assert len(news) > 0
        
        # Haiku should be multi-line, others single line
        assert '\n' in haiku
        assert '\n' not in sentence
        
        # Same image+vocab+format should always produce same output
        haiku2 = generator.generate(bark_path, "nature", "haiku")
        assert haiku == haiku2
        
    except FileNotFoundError:
        pytest.skip("Real bark image not available")

