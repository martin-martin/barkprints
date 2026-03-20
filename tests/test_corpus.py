"""Tests for corpus system."""

import numpy as np
import pytest

from barkprints.corpus import Corpus
from barkprints.corpus_loader import CorpusLoader


def test_corpus_creation():
    """Test creating a Corpus object with word-level fields."""
    vocabulary = ["bark.", "the", "tree"]
    word_embeddings = np.random.randn(3, 384)
    bigram_table = {"the": [("tree", 2), ("bark.", 1)]}
    start_words = ["the"]

    corpus = Corpus(
        name="test",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=start_words,
        metadata={"theme": "test"},
    )

    assert corpus.name == "test"
    assert len(corpus) == 3
    assert corpus.metadata["theme"] == "test"
    assert corpus.start_words == ["the"]
    assert "the" in corpus.bigram_table


def test_corpus_validation_mismatch():
    """Test corpus validation catches mismatched sizes."""
    vocabulary = ["a", "b", "c"]
    word_embeddings = np.random.randn(2, 384)  # Wrong size

    with pytest.raises(ValueError, match="Mismatch"):
        Corpus("test", vocabulary, word_embeddings, {}, [])


def test_corpus_validation_wrong_shape():
    """Test corpus validation catches wrong embedding shape."""
    vocabulary = ["a", "b"]
    word_embeddings = np.random.randn(2)  # 1D instead of 2D

    with pytest.raises(ValueError, match="2D"):
        Corpus("test", vocabulary, word_embeddings, {}, [])


def test_corpus_loader_list_available():
    """Test listing available corpora."""
    loader = CorpusLoader()
    corpora = loader.list_available()

    assert isinstance(corpora, list)
    assert "nature" in corpora
    assert "literature" in corpora


def test_corpus_loader_load_nature():
    """Test loading the nature corpus."""
    loader = CorpusLoader()
    corpus = loader.load("nature")

    assert corpus.name == "nature"
    assert len(corpus) > 0
    assert corpus.word_embeddings.shape[0] == len(corpus.vocabulary)
    assert corpus.word_embeddings.shape[1] > 0
    assert len(corpus.start_words) > 0
    assert len(corpus.bigram_table) > 0


def test_corpus_loader_nonexistent():
    """Test loading non-existent corpus raises error."""
    loader = CorpusLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_corpus_xyz")
