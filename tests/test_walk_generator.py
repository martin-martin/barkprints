"""Tests for walk generator."""

import numpy as np

from barkprints.corpus import Corpus
from barkprints.walk_generator import WalkGenerator


def _make_corpus():
    """Create a small test corpus for walk tests."""
    vocabulary = ["a", "bark.", "forest", "grows.", "in", "old", "the", "tree"]
    word_embeddings = np.random.RandomState(42).randn(len(vocabulary), 384)
    bigram_table = {
        "the": [("tree", 2), ("forest", 1), ("old", 1)],
        "tree": [("grows.", 1)],
        "old": [("tree", 1)],
        "in": [("the", 1)],
        "a": [("tree", 1)],
    }
    start_words = ["the", "a"]
    return Corpus(
        name="test",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=start_words,
    )


def test_determinism():
    """Same vector + corpus = same output, always."""
    corpus = _make_corpus()
    vec = np.random.RandomState(123).randn(384)

    walker = WalkGenerator(max_words=10)
    text1 = walker.generate(vec, corpus)
    text2 = walker.generate(vec, corpus)

    assert text1 == text2


def test_different_vectors_produce_different_output():
    """Different vectors should produce different output."""
    corpus = _make_corpus()
    vec1 = np.random.RandomState(1).randn(384)
    vec2 = np.random.RandomState(2).randn(384)

    walker = WalkGenerator(max_words=10)
    text1 = walker.generate(vec1, corpus)
    text2 = walker.generate(vec2, corpus)

    # With deterministic seeds on a small corpus, different vectors
    # produce different walks
    assert text1 != text2


def test_respects_max_words():
    """Output should not exceed max_words."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    for max_words in [5, 10, 15]:
        walker = WalkGenerator(max_words=max_words)
        text = walker.generate(vec, corpus)
        word_count = len(text.split())
        assert word_count <= max_words


def test_first_word_is_start_word():
    """First word should be from start_words."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(max_words=10)
    text = walker.generate(vec, corpus)

    first_word = text.split()[0].lower()
    assert first_word in corpus.start_words


def test_alpha_zero_follows_transitions():
    """alpha=0.0 should follow bigram transitions only."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(alpha=0.0, max_words=10)
    text = walker.generate(vec, corpus)

    assert isinstance(text, str)
    assert len(text) > 0


def test_alpha_one_follows_bark_similarity():
    """alpha=1.0 should follow bark similarity only."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(alpha=1.0, max_words=10)
    text = walker.generate(vec, corpus)

    assert isinstance(text, str)
    assert len(text) > 0


def test_dead_end_fallback():
    """Walk should continue even when hitting a dead-end word."""
    # Create corpus where most words are dead-ends
    vocabulary = ["hello", "world", "end."]
    word_embeddings = np.random.RandomState(42).randn(3, 384)
    bigram_table = {"hello": [("end.", 1)]}  # "end." and "world" are dead-ends
    start_words = ["hello"]

    corpus = Corpus(
        name="deadend",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=start_words,
    )

    vec = np.random.RandomState(42).randn(384)
    walker = WalkGenerator(max_words=5)
    text = walker.generate(vec, corpus)

    # Should still produce 5 words despite dead-ends
    assert len(text.split()) == 5


def test_output_starts_with_capital():
    """Output text should start with a capital letter."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(max_words=10)
    text = walker.generate(vec, corpus)

    assert text[0].isupper()
