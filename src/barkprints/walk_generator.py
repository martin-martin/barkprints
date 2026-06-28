"""Generate text by walking an n-gram model steered by bark image features."""

import numpy as np

from .corpus import Corpus

# Trailing characters that may sit after a sentence-ending mark, e.g. quotes or
# a closing parenthesis: 'word."' or 'word.)'.
_TRAILING = "\"')]}»”’"


class WalkGenerator:
    """Generate text via a deterministic n-gram walk steered by image features."""

    def __init__(self, alpha: float = 0.5, max_words: int = 20, min_words: int = 5):
        """Initialize walk generator.

        Args:
            alpha: Blend factor. 0.0 = pure n-gram coherence, 1.0 = pure bark similarity.
            max_words: Maximum number of words in output.
            min_words: Don't stop at a sentence end before this many words, so the
                walk can't terminate after one or two tokens.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive")
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        if min_words < 1:
            raise ValueError("min_words must be >= 1")
        self.alpha = alpha
        self.max_words = max_words
        self.min_words = min_words

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

    def _score_candidates(
        self,
        candidates: list[tuple[str, int]],
        feature_vector: np.ndarray,
        corpus: Corpus,
        vocab_index: dict[str, int],
    ) -> str:
        """Score bigram candidates by blending transition probability and bark similarity.

        Args:
            candidates: List of (next_word, count) tuples
            feature_vector: Current (possibly rolled) feature vector
            corpus: The corpus

        Returns:
            Selected next word
        """
        total_count = sum(count for _, count in candidates)

        best_score = -float("inf")
        best_word = candidates[0][0]

        for word, count in candidates:
            transition_prob = count / total_count
            idx = vocab_index.get(word)
            if idx is not None:
                sim = self._cosine_similarities(
                    feature_vector, corpus.word_embeddings[idx : idx + 1]
                )[0]
                bark_sim = (sim + 1) / 2  # Normalize [-1,1] to [0,1]
            else:
                bark_sim = 0.0

            score = (1 - self.alpha) * transition_prob + self.alpha * bark_sim
            if score > best_score:
                best_score = score
                best_word = word

        return best_word

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

        # Step 2: Walk
        for step in range(1, self.max_words):
            # Transform feature vector by rolling
            rolled = np.roll(feature_vector, step * stride)

            candidates = self._next_candidates(words, corpus)

            if candidates:
                # Score candidates by blended transition + bark similarity
                current_word = self._score_candidates(candidates, rolled, corpus, vocab_index)
            else:
                # Dead-end fallback: pick from entire vocabulary by bark similarity
                sims = self._cosine_similarities(rolled, corpus.word_embeddings)
                best_idx = int(np.argmax(sims))
                current_word = corpus.vocabulary[best_idx]

            words.append(current_word)

            # Stop at a natural sentence end once we have enough words.
            if len(words) >= self.min_words and self._is_sentence_end(current_word):
                break

        # Post-processing: capitalize first character
        text = " ".join(words)
        text = text[0].upper() + text[1:] if text else ""

        return text
