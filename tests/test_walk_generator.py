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


def test_trigram_context_overrides_bigram():
    """When a two-word context is known, the trigram table steers the next word."""
    vocabulary = ["a", "b", "x", "y"]
    word_embeddings = np.random.RandomState(0).randn(len(vocabulary), 384)
    # Bigram of "b" alone would pick "x"; the "a b" trigram picks "y" instead.
    bigram_table = {"a": [("b", 1)], "b": [("x", 1)]}
    trigram_table = {"a b": [("y", 1)]}
    corpus = Corpus(
        name="tri",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["a"],
        trigram_table=trigram_table,
    )

    vec = np.random.RandomState(1).randn(384)
    # alpha=0 -> pure transitions; min_words high so it doesn't stop early.
    walker = WalkGenerator(alpha=0.0, max_words=3, min_words=5)
    words = walker.generate(vec, corpus).lower().split()

    assert words == ["a", "b", "y"]


def test_stops_at_sentence_end():
    """The walk ends at a sentence-final token once past min_words."""
    vocabulary = ["grows", "old", "the", "tree."]
    word_embeddings = np.random.RandomState(0).randn(len(vocabulary), 384)
    bigram_table = {"the": [("old", 1)], "old": [("tree.", 1)], "tree.": [("grows", 1)]}
    corpus = Corpus(
        name="stop",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["the"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=10, min_words=3)
    text = walker.generate(vec, corpus)

    # the -> old -> tree. : three words, then stop on the sentence-ending token.
    assert text.split() == ["The", "old", "tree."]


def test_cycle_is_escaped_not_looped():
    """A bigram cycle can only be re-entered while fresh edges remain, so the
    walk escapes instead of looping to max_words."""
    # "the -> most -> of -> the" is a cycle the old greedy walk looped on
    # forever; "of" also offers an escape ("bark.").
    vocabulary = ["bark.", "most", "of", "the"]
    word_embeddings = np.random.RandomState(0).randn(len(vocabulary), 384)
    bigram_table = {
        "the": [("most", 3)],
        "most": [("of", 3)],
        "of": [("the", 3), ("bark.", 1)],
    }
    corpus = Corpus(
        name="cycle",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["the"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=12, min_words=2)
    words = walker.generate(vec, corpus).lower().split()

    # Second visit to "of" has (of, the) marked used, so "bark." is the only
    # fresh candidate and the walk ends there — well short of max_words.
    assert words[-1] == "bark."
    assert len(words) < 12


def test_forced_single_path_may_repeat():
    """When every continuation repeats a used edge, the walk still continues."""
    # Pure two-word cycle with no escape: a -> b -> a -> b ...
    vocabulary = ["a", "b"]
    word_embeddings = np.random.RandomState(0).randn(2, 384)
    bigram_table = {"a": [("b", 1)], "b": [("a", 1)]}
    corpus = Corpus(
        name="forced",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["a"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=6, min_words=6)
    words = walker.generate(vec, corpus).lower().split()

    assert words == ["a", "b", "a", "b", "a", "b"]


def test_word_reuse_decay_prefers_fresh_words():
    """A heavily reused word loses to a fresh alternative despite higher counts."""
    # "b" is the common follower of "a" and ties with "d" after "c" — but by
    # the time the walk reaches "c", "b" has been used once, so its halved
    # score (0.5 * 0.5 = 0.25) loses to the fresh "d" (0.5) at alpha=0.
    vocabulary = ["a", "b", "c", "d"]
    word_embeddings = np.random.RandomState(0).randn(4, 384)
    bigram_table = {
        "a": [("b", 9), ("c", 1)],
        "b": [("a", 1)],
        "c": [("b", 1), ("d", 1)],
    }
    corpus = Corpus(
        name="decay",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["a"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=6, min_words=6)
    words = walker.generate(vec, corpus).lower().split()

    # Walk: a -> b (0.9 beats 0.1). b -> a (forced). a -> c: the a->b edge is
    # used, so "c" is the only fresh candidate. c -> d: b has been used once,
    # so the fresh "d" wins.
    assert words[:5] == ["a", "b", "a", "c", "d"]
    """A sentence end before min_words does not stop the walk."""
    vocabulary = ["ab.", "cd", "de", "the"]
    word_embeddings = np.random.RandomState(0).randn(len(vocabulary), 384)
    bigram_table = {"the": [("ab.", 1)], "ab.": [("cd", 1)], "cd": [("de", 1)]}
    corpus = Corpus(
        name="floor",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["the"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=4, min_words=4)
    words = walker.generate(vec, corpus).split()

    # "ab." at position 2 is a sentence end but below min_words=4, so the walk
    # continues to the full length.
    assert len(words) == 4
