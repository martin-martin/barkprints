"""Tests for corpus system."""

import numpy as np
import pytest

from barkprints.corpus import Corpus
from barkprints.corpus_loader import CorpusLoader


def test_corpus_creation():
    """Test creating a Corpus object."""
    sentences = ["First sentence.", "Second sentence."]
    embeddings = np.random.randn(2, 384)
    
    corpus = Corpus(
        name="test",
        sentences=sentences,
        embeddings=embeddings,
        metadata={"theme": "test"}
    )
    
    assert corpus.name == "test"
    assert len(corpus) == 2
    assert corpus.metadata["theme"] == "test"


def test_corpus_validation_mismatch():
    """Test corpus validation catches mismatched sizes."""
    sentences = ["First.", "Second.", "Third."]
    embeddings = np.random.randn(2, 384)  # Wrong size
    
    with pytest.raises(ValueError, match="Mismatch"):
        Corpus("test", sentences, embeddings)


def test_corpus_validation_wrong_shape():
    """Test corpus validation catches wrong embedding shape."""
    sentences = ["First.", "Second."]
    embeddings = np.random.randn(2)  # 1D instead of 2D
    
    with pytest.raises(ValueError, match="2D"):
        Corpus("test", sentences, embeddings)


def test_corpus_loader_list_available():
    """Test listing available corpora."""
    loader = CorpusLoader()
    corpora = loader.list_available()
    
    # Should include our built-in corpora
    assert isinstance(corpora, list)
    assert "nature" in corpora
    assert "literature" in corpora


def test_corpus_loader_load_nature():
    """Test loading the nature corpus."""
    loader = CorpusLoader()
    corpus = loader.load("nature")
    
    assert corpus.name == "nature"
    assert len(corpus) > 0
    assert corpus.embeddings.shape[0] == len(corpus.sentences)
    assert corpus.embeddings.shape[1] > 0  # Has embeddings


def test_corpus_loader_nonexistent():
    """Test loading non-existent corpus raises error."""
    loader = CorpusLoader()
    
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_corpus_xyz")

