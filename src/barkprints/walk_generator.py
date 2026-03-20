"""Generate text by walking a bigram model steered by bark image features."""

import numpy as np

from .corpus import Corpus


class WalkGenerator:
    """Generate text via a deterministic bigram walk steered by image features."""

    def __init__(self, alpha: float = 0.5, max_words: int = 20):
        """Initialize walk generator.

        Args:
            alpha: Blend factor. 0.0 = pure bigram coherence, 1.0 = pure bark similarity.
            max_words: Maximum number of words in output.
        """
        self.alpha = alpha
        self.max_words = max_words

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

    def _word_index(self, word: str, corpus: Corpus) -> int | None:
        """Find the index of a word in the vocabulary."""
        try:
            return corpus.vocabulary.index(word)
        except ValueError:
            return None

    def _score_candidates(
        self,
        candidates: list[tuple[str, int]],
        feature_vector: np.ndarray,
        corpus: Corpus,
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
            idx = self._word_index(word, corpus)
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
        stride = len(feature_vector) // self.max_words

        # Step 1: Pick first word from start_words by bark similarity
        start_indices = []
        for word in corpus.start_words:
            idx = self._word_index(word, corpus)
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

            candidates = corpus.bigram_table.get(current_word)

            if candidates:
                # Score candidates by blended transition + bark similarity
                current_word = self._score_candidates(candidates, rolled, corpus)
            else:
                # Dead-end fallback: pick from entire vocabulary by bark similarity
                sims = self._cosine_similarities(rolled, corpus.word_embeddings)
                best_idx = int(np.argmax(sims))
                current_word = corpus.vocabulary[best_idx]

            words.append(current_word)

        # Post-processing: capitalize first character
        text = " ".join(words)
        text = text[0].upper() + text[1:] if text else ""

        return text
