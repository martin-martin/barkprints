"""Corpus data structures for word-level generative text."""

import numpy as np


class Corpus:
    """Represents a text corpus with word-level embeddings and bigram transitions."""

    def __init__(
        self,
        name: str,
        vocabulary: list[str],
        word_embeddings: np.ndarray,
        bigram_table: dict[str, list[tuple[str, int]]],
        start_words: list[str],
        metadata: dict | None = None,
    ):
        """Initialize corpus.

        Args:
            name: Name of the corpus
            vocabulary: Unique words from corpus (sorted)
            word_embeddings: (V, 384) word embedding matrix
            bigram_table: word -> [(next_word, count), ...]
            start_words: Words that begin sentences
            metadata: Additional metadata (theme, source, etc.)
        """
        self.name = name
        self.vocabulary = vocabulary
        self.word_embeddings = word_embeddings
        self.bigram_table = bigram_table
        self.start_words = start_words
        self.metadata = metadata or {}

        # Validate
        if len(vocabulary) != len(word_embeddings):
            raise ValueError(
                f"Mismatch: {len(vocabulary)} words but {len(word_embeddings)} embeddings"
            )

        if word_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array, got shape {word_embeddings.shape}"
            )

    def __len__(self) -> int:
        """Return number of words in vocabulary."""
        return len(self.vocabulary)

    def __repr__(self) -> str:
        return f"Corpus(name='{self.name}', vocab_size={len(self)}, theme='{self.metadata.get('theme', 'unknown')}')"
