"""Tests for vocabulary loading and management."""

import json
import pytest

from barkprints.vocabulary import Vocabulary
from barkprints.vocabulary_loader import VocabularyLoader


def test_vocabulary_creation():
    """Test creating a Vocabulary object."""
    words = {"1": ["test", "word"], "2": ["hello"]}
    vocab = Vocabulary(
        name="test",
        theme="test theme",
        output_format="haiku",
        words=words
    )
    
    assert vocab.name == "test"
    assert vocab.theme == "test theme"
    assert vocab.output_format == "haiku"
    assert vocab.words == words


def test_vocabulary_loader_initialization(haiku_vocabulary):
    """Test VocabularyLoader initialization."""
    vocab_dir, _ = haiku_vocabulary
    loader = VocabularyLoader(vocab_dir)
    
    assert loader.vocabularies_dir == vocab_dir


def test_load_vocabulary(haiku_vocabulary):
    """Test loading a vocabulary file."""
    vocab_dir, vocab_name = haiku_vocabulary
    loader = VocabularyLoader(vocab_dir)
    
    vocab = loader.load(vocab_name)
    
    assert isinstance(vocab, Vocabulary)
    assert vocab.name == vocab_name
    assert vocab.output_format == "haiku"
    assert "1" in vocab.words
    assert len(vocab.words["1"]) > 0


def test_load_nonexistent_vocabulary(temp_dir):
    """Test loading a vocabulary that doesn't exist."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir()
    loader = VocabularyLoader(vocab_dir)
    
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")


def test_load_invalid_vocabulary(temp_dir):
    """Test loading an invalid vocabulary file."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir()
    
    # Create invalid vocabulary (missing required fields)
    invalid_vocab = {"name": "invalid"}
    vocab_file = vocab_dir / "invalid.json"
    with open(vocab_file, 'w') as f:
        json.dump(invalid_vocab, f)
    
    loader = VocabularyLoader(vocab_dir)
    
    with pytest.raises(ValueError):
        loader.load("invalid")


def test_list_available_vocabularies(haiku_vocabulary, commentary_vocabulary):
    """Test listing available vocabularies."""
    vocab_dir, _ = haiku_vocabulary
    loader = VocabularyLoader(vocab_dir)
    
    available = loader.list_available()
    
    assert "test_haiku" in available
    assert "test_commentary" in available
    assert len(available) == 2


def test_list_available_empty_directory(temp_dir):
    """Test listing vocabularies in an empty directory."""
    vocab_dir = temp_dir / "vocabularies"
    vocab_dir.mkdir()
    loader = VocabularyLoader(vocab_dir)
    
    available = loader.list_available()
    
    assert available == []

