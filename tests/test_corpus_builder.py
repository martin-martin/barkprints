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
        ["the", "tree", "grows", "tall"],
        ["wind", "blows", "gently"],
    ]


def test_tokenize_strips_punctuation(builder):
    """Punctuation is split off tokens, so variants merge in the vocabulary."""
    sentences = ["Hello, world! How are you?"]
    result = builder.tokenize(sentences)

    assert result == [["hello", "world", "how", "are", "you"]]


def test_tokenize_keeps_internal_apostrophes_and_hyphens(builder):
    """Words like don't and old-growth survive as single tokens."""
    sentences = ["Don't fell old-growth trees."]
    result = builder.tokenize(sentences)

    assert result == [["don't", "fell", "old-growth", "trees"]]


def test_tokenize_strips_gutenberg_markup(builder):
    """Underscore italics markup (_quercus_) cleans away."""
    sentences = ["The oak (_Quercus alba_) stands, alone,—but proud."]
    result = builder.tokenize(sentences)

    assert result == [["the", "oak", "quercus", "alba", "stands", "alone", "but", "proud"]]


def test_tokenize_handles_umlauts(builder):
    """Non-ASCII letters are part of words, not punctuation."""
    sentences = ["Die Bäume wachsen über größere Höhen."]
    result = builder.tokenize(sentences)

    assert result == [["die", "bäume", "wachsen", "über", "größere", "höhen"]]


def test_end_word_statistics(builder):
    """End words record how often each word closed a sentence."""
    text = "The tree grows. Wind shakes the tree. The tree grows tall."
    corpus = builder.build_corpus(text, "test")

    # "tree" ends 1 of its 3 occurrences; "grows" 1 of 2; "tall" 1 of 1.
    assert corpus.end_words["tree"] == (1, 3)
    assert corpus.end_words["grows"] == (1, 2)
    assert corpus.end_words["tall"] == (1, 1)
    assert "wind" not in corpus.end_words


def test_bigram_table_accuracy(builder):
    """Bigram table should accurately count word transitions."""
    text = "The tree grows. The tree stands. The wind blows."
    corpus = builder.build_corpus(text, "test")

    # "the" should transition to "tree" (2x) and "wind" (1x)
    the_bigrams = dict(corpus.bigram_table.get("the", []))
    assert the_bigrams.get("tree", 0) == 2
    assert the_bigrams.get("wind", 0) == 1


def test_trigram_table_accuracy(builder):
    """Trigram table should count two-word-context transitions."""
    text = "The tree grows tall. The tree grows fast. The tree stands still."
    corpus = builder.build_corpus(text, "test")

    # Context "the tree" is followed by "grows" (2x) and "stands" (1x).
    ctx = dict(corpus.trigram_table.get("the tree", []))
    assert ctx.get("grows", 0) == 2
    assert ctx.get("stands", 0) == 1


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
        assert loaded.trigram_table == corpus.trigram_table
        assert loaded.end_words == corpus.end_words
        np.testing.assert_array_almost_equal(
            loaded.word_embeddings, corpus.word_embeddings
        )


def test_migrate_legacy_corpus(builder):
    """Migration merges punctuation variants and derives end statistics."""
    from barkprints.corpus import Corpus
    from barkprints.corpus_migrator import migrate_corpus

    # A legacy-style corpus as the old whitespace tokenizer would have built
    # it from: "The tree grows. The tree stands, alone." — attached
    # punctuation makes 'grows.' / 'stands,' / 'alone.' distinct tokens.
    vocabulary = ["alone.", "grows.", "stands,", "the", "tree"]
    word_embeddings = np.zeros((5, 4))
    legacy = Corpus(
        name="legacy",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table={
            "the": [("tree", 2)],
            "tree": [("grows.", 1), ("stands,", 1)],
            "stands,": [("alone.", 1)],
        },
        start_words=["the"],
    )

    migrated = migrate_corpus(legacy, builder.model)

    assert migrated.vocabulary == ["alone", "grows", "stands", "the", "tree"]
    assert dict(migrated.bigram_table["tree"]) == {"grows": 1, "stands": 1}
    assert dict(migrated.bigram_table["stands"]) == {"alone": 1}
    # 'grows.' and 'alone.' carried terminal punctuation -> end words.
    assert migrated.end_words["grows"] == (1, 1)
    assert migrated.end_words["alone"] == (1, 1)
    assert "stands" not in migrated.end_words
    assert migrated.start_words == ["the"]
    assert len(migrated.word_embeddings) == len(migrated.vocabulary)
    assert migrated.metadata["cleaned"] is True
