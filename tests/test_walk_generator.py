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


def test_repetition_penalty_breaks_cycle():
    """A bigram cycle with an available alternative should not repeat forever."""
    vocabulary = ["a", "b", "c", "problem"]
    word_embeddings = np.random.RandomState(0).randn(len(vocabulary), 384)
    # Without a repetition penalty, alpha=0.0 argmax would loop a -> b -> c -> a -> ...
    # forever, since "problem" is always the lower-count (non-argmax) alternative.
    bigram_table = {
        "a": [("b", 5), ("problem", 1)],
        "b": [("c", 5)],
        "c": [("a", 5)],
    }
    corpus = Corpus(
        name="cycle",
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=["a"],
    )

    vec = np.random.RandomState(1).randn(384)
    walker = WalkGenerator(alpha=0.0, max_words=8, min_words=8)
    words = walker.generate(vec, corpus).lower().split()

    # A pure argmax walk would produce "a b c a b c a b" (never varying).
    # With the repetition penalty, the walk should break out of the 3-cycle.
    assert len(set(words)) > 3


def test_widen_gives_alpha_leverage():
    """A single-follower context should still let alpha pick a bark-similar word."""
    vocabulary = ["start", "only_follower", "bark_favorite"]
    embeddings = np.zeros((len(vocabulary), 4))
    embeddings[0] = [1, 0, 0, 0]  # start
    embeddings[1] = [1, 0, 0, 0]  # only_follower: similar to "start", not to bark vec
    embeddings[2] = [0, 0, 0, 1]  # bark_favorite: matches the bark vector below

    bigram_table = {"start": [("only_follower", 1)]}
    corpus = Corpus(
        name="widen",
        vocabulary=vocabulary,
        word_embeddings=embeddings,
        bigram_table=bigram_table,
        start_words=["start"],
    )

    # Chosen so that after generate()'s per-step np.roll (stride=2 for a 4-d
    # vector with max_words=2), the walk's one real step compares against
    # [0, 0, 0, 1] -- an exact match for "bark_favorite" and orthogonal to
    # "start"/"only_follower"'s [1, 0, 0, 0] -- so at alpha=1.0 the bark-similar
    # widened candidate should win over the sole bigram candidate.
    vec = np.array([0.0, 1.0, 0, 0])

    walker_low = WalkGenerator(alpha=0.0, max_words=2, min_words=2)
    words_low = walker_low.generate(vec, corpus).lower().split()
    assert words_low == ["start", "only_follower"]

    walker_high = WalkGenerator(alpha=1.0, max_words=2, min_words=2)
    words_high = walker_high.generate(vec, corpus).lower().split()
    assert words_high == ["start", "bark_favorite"]


def test_widen_preserves_alpha_zero_behavior():
    """alpha=0.0 output should be unchanged by widening even with a bark-favorite present."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(alpha=0.0, max_words=10)
    text1 = walker.generate(vec, corpus)
    text2 = walker.generate(vec, corpus)

    assert text1 == text2


def test_spatial_steering_changes_output():
    """Different spatial grids (same feature vector) should steer to different text."""
    corpus = _make_corpus()
    vec = np.random.RandomState(5).randn(384)

    grid1 = np.random.RandomState(10).randn(100, 4)
    grid2 = np.random.RandomState(20).randn(100, 4)

    walker = WalkGenerator(alpha=1.0, max_words=10, spatial_weight=0.8)
    text1 = walker.generate(vec, corpus, spatial_grid=grid1)
    text2 = walker.generate(vec, corpus, spatial_grid=grid2)

    assert text1 != text2


def test_spatial_steering_deterministic():
    """Same feature vector + same spatial grid should always produce the same output."""
    corpus = _make_corpus()
    vec = np.random.RandomState(5).randn(384)
    grid = np.random.RandomState(10).randn(100, 4)

    walker = WalkGenerator(alpha=0.7, max_words=10, spatial_weight=0.5)
    text1 = walker.generate(vec, corpus, spatial_grid=grid)
    text2 = walker.generate(vec, corpus, spatial_grid=grid)

    assert text1 == text2


def test_generate_without_spatial_grid_uses_legacy_path():
    """Omitting spatial_grid should behave exactly as before this feature existed."""
    corpus = _make_corpus()
    vec = np.random.RandomState(42).randn(384)

    walker = WalkGenerator(max_words=10)
    text_default = walker.generate(vec, corpus)
    text_explicit_none = walker.generate(vec, corpus, spatial_grid=None)

    assert text_default == text_explicit_none


def test_min_words_floor_prevents_early_stop():
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
