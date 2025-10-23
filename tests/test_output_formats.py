"""Tests for output format plugins."""

from random import Random

import pytest

from barkprints.vocabulary import Vocabulary
from barkprints.output_formats import HaikuFormat, CommentaryFormat, SentenceFormat
from barkprints.format_loader import FormatLoader


def test_haiku_format_name():
    """Test haiku format name."""
    fmt = HaikuFormat()
    assert fmt.format_name == "haiku"


def test_commentary_format_name():
    """Test commentary format name."""
    fmt = CommentaryFormat()
    assert fmt.format_name == "commentary"


def test_sentence_format_name():
    """Test sentence format name."""
    fmt = SentenceFormat()
    assert fmt.format_name == "sentence"


def test_haiku_generation_with_valid_vocabulary():
    """Test haiku generation with properly structured vocabulary."""
    words = {
        "1": ["wind", "rain", "sun", "tree", "leaf", "bird", "stone", "wood"],
        "2": ["sunset", "morning", "winter", "summer", "forest", "moonlight"],
        "3": ["beautiful", "whispering", "evergreen", "wandering", "colorful"],
        "4": ["awakening", "meditation", "eternal spring"],
        "5": ["metamorphosis", "illumination"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="haiku",
        words=words
    )
    
    fmt = HaikuFormat()
    rng = Random(42)
    
    haiku = fmt.generate(vocab, rng)
    
    # Check that we got a string with 3 lines
    lines = haiku.split('\n')
    assert len(lines) == 3
    
    # Check that each line has content
    for line in lines:
        assert len(line.strip()) > 0


def test_haiku_determinism():
    """Test that haiku generation is deterministic with same seed."""
    words = {
        "1": ["wind", "rain", "sun", "tree", "leaf", "bird"],
        "2": ["sunset", "morning", "winter", "summer"],
        "3": ["beautiful", "whispering", "evergreen"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="haiku",
        words=words
    )
    
    fmt = HaikuFormat()
    
    # Generate with same seed twice
    haiku1 = fmt.generate(vocab, Random(42))
    haiku2 = fmt.generate(vocab, Random(42))
    
    assert haiku1 == haiku2


def test_haiku_with_invalid_vocabulary():
    """Test haiku generation with non-syllable vocabulary raises error."""
    words = {
        "subjects": ["test"],
        "verbs": ["do"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="commentary",
        words=words
    )
    
    fmt = HaikuFormat()
    rng = Random(42)
    
    with pytest.raises(ValueError, match="syllable-organized"):
        fmt.generate(vocab, rng)


def test_commentary_generation():
    """Test commentary generation."""
    words = {
        "subjects": ["the world", "society"],
        "verbs": ["evolves", "changes"],
        "descriptors": ["rapidly", "slowly"],
        "context": ["today", "now"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="commentary",
        words=words
    )
    
    fmt = CommentaryFormat()
    rng = Random(42)
    
    text = fmt.generate(vocab, rng)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.endswith('.')


def test_commentary_determinism():
    """Test that commentary generation is deterministic."""
    words = {
        "subjects": ["the world"],
        "verbs": ["evolves"],
        "descriptors": ["rapidly"],
        "context": ["today"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="commentary",
        words=words
    )
    
    fmt = CommentaryFormat()
    
    text1 = fmt.generate(vocab, Random(42))
    text2 = fmt.generate(vocab, Random(42))
    
    assert text1 == text2


def test_sentence_generation():
    """Test sentence generation."""
    words = {
        "1": ["word", "test", "life", "time"]
    }
    
    vocab = Vocabulary(
        name="test",
        theme="test",
        output_format="sentence",
        words=words
    )
    
    fmt = SentenceFormat()
    rng = Random(42)
    
    text = fmt.generate(vocab, rng)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert text[0].isupper()  # First letter capitalized
    assert text.endswith('.')


def test_format_loader_registration():
    """Test that format loader registers default formats."""
    loader = FormatLoader()
    
    available = loader.list_available()
    
    assert "haiku" in available
    assert "commentary" in available
    assert "sentence" in available


def test_format_loader_get():
    """Test getting a format from the loader."""
    loader = FormatLoader()
    
    fmt = loader.get("haiku")
    
    assert isinstance(fmt, HaikuFormat)
    assert fmt.format_name == "haiku"


def test_format_loader_get_nonexistent():
    """Test getting a format that doesn't exist."""
    loader = FormatLoader()
    
    with pytest.raises(KeyError):
        loader.get("nonexistent")

