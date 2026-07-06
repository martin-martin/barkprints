"""Generate text by walking an n-gram model steered by bark image features."""

import numpy as np

from .corpus import Corpus

# Trailing characters that may sit after a sentence-ending mark, e.g. quotes or
# a closing parenthesis: 'word."' or 'word.)'.
_TRAILING = "\"')]}»”’"


class WalkGenerator:
    """Generate text via a deterministic n-gram walk steered by image features."""

    def __init__(
        self,
        alpha: float = 0.5,
        max_words: int = 20,
        min_words: int = 5,
        end_threshold: float = 0.5,
    ):
        """Initialize walk generator.

        Args:
            alpha: Blend factor. 0.0 = pure n-gram coherence, 1.0 = pure bark similarity.
            max_words: Maximum number of words in output.
            min_words: Don't stop at a sentence end before this many words, so the
                walk can't terminate after one or two tokens.
            end_threshold: On corpora with end-word statistics, a word may end
                the poem when it closed a sentence in at least this fraction of
                its corpus occurrences.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive")
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        if min_words < 1:
            raise ValueError("min_words must be >= 1")
        if not (0.0 < end_threshold <= 1.0):
            raise ValueError("end_threshold must be in (0.0, 1.0]")
        self.alpha = alpha
        self.max_words = max_words
        self.min_words = min_words
        self.end_threshold = end_threshold

    @staticmethod
    def _is_sentence_end(word: str) -> bool:
        """True if a token ends a sentence (ends in . ! ? with real letters).

        Guards against stopping on ordinals or abbreviations like '20.' or 'z.'
        by requiring at least two alphabetic characters in the token.
        """
        stripped = word.rstrip(_TRAILING)
        if not stripped or stripped[-1] not in ".!?":
            return False
        return sum(c.isalpha() for c in stripped) >= 2

    def _ends_sentence(self, word: str, corpus: Corpus) -> bool:
        """Whether the walk may stop on this word.

        Cleaned corpora carry end-word statistics (punctuation is stripped from
        tokens): a word qualifies when it closed a sentence in at least
        end_threshold of its occurrences. Legacy corpora signal sentence ends
        via punctuation attached to the token.
        """
        if corpus.end_words is not None:
            entry = corpus.end_words.get(word)
            if entry is None:
                return False
            end_count, total_count = entry
            return total_count > 0 and end_count / total_count >= self.end_threshold
        return self._is_sentence_end(word)

    @staticmethod
    def _next_candidates(
        words: list[str], corpus: Corpus
    ) -> list[tuple[str, int]] | None:
        """Pick the candidate next words, preferring trigram context over bigram.

        Uses the two-word context ('prev current') when the corpus has a trigram
        table and that context is known; otherwise falls back to the bigram table
        for the current word. Returns None when neither offers a continuation.
        """
        current = words[-1]
        if corpus.trigram_table is not None and len(words) >= 2:
            key = f"{words[-2]} {current}"
            trigram = corpus.trigram_table.get(key)
            if trigram:
                return trigram
        return corpus.bigram_table.get(current)

    @staticmethod
    def _cosine_similarities(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between a vector and each row of a matrix.

        Args:
            vec: (D,) query vector
            matrix: (N, D) matrix of vectors

        Returns:
            (N,) array of cosine similarities
        """
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return np.zeros(matrix.shape[0])

        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms = np.where(row_norms == 0, 1e-10, row_norms)

        return matrix @ vec / (row_norms * vec_norm)

    # Each prior use of a word halves its score, so the greedy walk is pushed
    # toward the next-most-similar fresh word instead of circling favorites.
    WORD_REUSE_DECAY = 0.5

    @staticmethod
    def _percentile_ranks(values: np.ndarray) -> np.ndarray:
        """Map values to percentile ranks in (0, 1); ties share their mean rank.

        Raw transition probabilities and bark cosine similarities live on very
        different scales (a dominant follower has probability ~1.0 while
        similarities cluster in a narrow band), which made alpha behave like a
        cliff instead of a dial. Ranking both signals within the candidate set
        puts them on the same scale, so blending is meaningful at every alpha.
        """
        n = len(values)
        ranks = np.empty(n)
        ranks[np.argsort(values, kind="stable")] = np.arange(n, dtype=float)
        for value in np.unique(values):
            mask = values == value
            ranks[mask] = ranks[mask].mean()
        # (rank + 0.5) / n keeps every rank strictly positive, so the reuse
        # decay still differentiates candidates at the bottom of the order.
        return (ranks + 0.5) / n

    def _score_candidates(
        self,
        candidates: list[tuple[str, int]],
        feature_vector: np.ndarray,
        corpus: Corpus,
        vocab_index: dict[str, int],
        word_use: dict[str, int],
    ) -> str:
        """Score candidates by blending transition frequency and bark similarity.

        Both signals are converted to percentile ranks within the candidate
        set before blending, so alpha trades them off smoothly.

        Args:
            candidates: List of (next_word, count) tuples
            feature_vector: Current steering feature vector
            corpus: The corpus
            vocab_index: word -> row index into corpus.word_embeddings
            word_use: How often each word already appears in this walk

        Returns:
            Selected next word
        """
        counts = np.array([count for _, count in candidates], dtype=float)
        sims = np.full(len(candidates), -1.0)
        for k, (word, _) in enumerate(candidates):
            idx = vocab_index.get(word)
            if idx is not None:
                sims[k] = self._cosine_similarities(
                    feature_vector, corpus.word_embeddings[idx : idx + 1]
                )[0]

        coherence_rank = self._percentile_ranks(counts)
        bark_rank = self._percentile_ranks(sims)
        scores = (1 - self.alpha) * coherence_rank + self.alpha * bark_rank
        scores *= np.array(
            [self.WORD_REUSE_DECAY ** word_use.get(word, 0) for word, _ in candidates]
        )

        return candidates[int(np.argmax(scores))][0]

    def generate(self, feature_vector: np.ndarray, corpus: Corpus) -> str:
        """Generate text by walking the bigram model steered by the feature vector.

        Args:
            feature_vector: (D,) image feature vector
            corpus: Corpus with vocabulary, embeddings, bigram table, start words

        Returns:
            Generated text string
        """
        stride = max(1, len(feature_vector) // self.max_words)
        vocab_index = {w: i for i, w in enumerate(corpus.vocabulary)}

        # Step 1: Pick first word from start_words by bark similarity
        start_indices = []
        for word in corpus.start_words:
            idx = vocab_index.get(word)
            if idx is not None:
                start_indices.append((word, idx))

        if not start_indices:
            # Fallback: pick from entire vocabulary
            sims = self._cosine_similarities(feature_vector, corpus.word_embeddings)
            best_idx = int(np.argmax(sims))
            current_word = corpus.vocabulary[best_idx]
        else:
            start_embeddings = np.array(
                [corpus.word_embeddings[idx] for _, idx in start_indices]
            )
            sims = self._cosine_similarities(feature_vector, start_embeddings)
            best = int(np.argmax(sims))
            current_word = start_indices[best][0]

        words = [current_word]
        word_use: dict[str, int] = {current_word: 1}
        used_bigrams: set[tuple[str, str]] = set()

        # Step 2: Walk
        for step in range(1, self.max_words):
            # Transform feature vector by rolling
            rolled = np.roll(feature_vector, step * stride)

            candidates = self._next_candidates(words, corpus)

            if candidates:
                # Never re-take an edge already walked in this poem, unless the
                # path is forced (every candidate would repeat a used bigram).
                previous = words[-1]
                fresh = [
                    (word, count)
                    for word, count in candidates
                    if (previous, word) not in used_bigrams
                ]
                if fresh:
                    candidates = fresh
                # Score candidates by blended transition + bark similarity
                current_word = self._score_candidates(
                    candidates, rolled, corpus, vocab_index, word_use
                )
                used_bigrams.add((previous, current_word))
            else:
                # Dead-end fallback: pick from entire vocabulary by bark
                # similarity, decayed for words this walk already used.
                sims = (self._cosine_similarities(rolled, corpus.word_embeddings) + 1) / 2
                for word, uses in word_use.items():
                    idx = vocab_index.get(word)
                    if idx is not None:
                        sims[idx] *= self.WORD_REUSE_DECAY**uses
                best_idx = int(np.argmax(sims))
                current_word = corpus.vocabulary[best_idx]

            words.append(current_word)
            word_use[current_word] = word_use.get(current_word, 0) + 1

            # Stop at a natural sentence end once we have enough words.
            if len(words) >= self.min_words and self._ends_sentence(current_word, corpus):
                natural_stop = True
                break
        else:
            natural_stop = self.max_words == 1 or self._ends_sentence(
                current_word, corpus
            )

        # Post-processing: capitalize first character
        text = " ".join(words)
        text = text[0].upper() + text[1:] if text else ""

        # Cleaned corpora produce bare words; close the poem with a period
        # when it ended on a natural sentence end, an ellipsis when it was
        # cut off at max_words mid-phrase.
        if corpus.end_words is not None and text:
            text += "." if natural_stop else "…"

        return text
