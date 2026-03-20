"""Tests for corpus builder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from barkprints.corpus_builder import CorpusBuilder, save_corpus
from barkprints.corpus_loader import CorpusLoader


@pytest.fixture(scope="module")
def builder():
    """Create a shared builder (loading model is expensive)."""
    return CorpusBuilder()


def test_tokenize(builder):
    """Test tokenization correctness."""
    sentences = ["The tree grows tall.", "Wind blows gently."]
    result = builder.tokenize(sentences)

    assert result == [
        ["the", "tree", "grows", "tall."],
        ["wind", "blows", "gently."],
    ]


def test_tokenize_preserves_punctuation(builder):
    """Punctuation stays attached to words."""
    sentences = ["Hello, world! How are you?"]
    result = builder.tokenize(sentences)

    assert result == [["hello,", "world!", "how", "are", "you?"]]


def test_bigram_table_accuracy(builder):
    """Bigram table should accurately count word transitions."""
    text = "The tree grows. The tree stands. The wind blows."
    corpus = builder.build_corpus(text, "test")

    # "the" should transition to "tree" (2x) and "wind" (1x)
    the_bigrams = dict(corpus.bigram_table.get("the", []))
    assert the_bigrams.get("tree", 0) == 2
    assert the_bigrams.get("wind", 0) == 1


def test_start_words_identified(builder):
    """Start words should be the first word of each sentence."""
    text = "The tree grows. Wind blows hard. The forest stands."
    corpus = builder.build_corpus(text, "test")

    assert "the" in corpus.start_words
    assert "wind" in corpus.start_words


def test_vocabulary_sorted(builder):
    """Vocabulary should be sorted for determinism."""
    text = "Zebra runs fast. Apple grows slowly. Mango falls down."
    corpus = builder.build_corpus(text, "test")

    assert corpus.vocabulary == sorted(corpus.vocabulary)


def test_round_trip_save_load(builder):
    """Save and load should preserve all data."""
    text = "The forest grows. Trees stand tall. The wind whispers."
    corpus = builder.build_corpus(text, "roundtrip", {"theme": "test"})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.npz"
        save_corpus(corpus, path)

        loader = CorpusLoader(corpora_dir=Path(tmpdir))
        loaded = loader.load("test")

        assert loaded.vocabulary == corpus.vocabulary
        assert loaded.start_words == corpus.start_words
        assert loaded.bigram_table == corpus.bigram_table
        np.testing.assert_array_almost_equal(
            loaded.word_embeddings, corpus.word_embeddings
        )
