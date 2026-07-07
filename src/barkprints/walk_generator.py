"""Generate text by walking an n-gram model steered by bark image features."""

import numpy as np

from .corpus import Corpus

# Trailing characters that may sit after a sentence-ending mark, e.g. quotes or
# a closing parenthesis: 'word."' or 'word.)'.
_TRAILING = "\"')]}»”’"


class WalkGenerator:
    """Generate text via a deterministic n-gram walk steered by image features."""

    # How much a repeated (context, next-word) pair is penalized per prior
    # occurrence. The blended score lives in [0, 1], so one repeat (0.5) is
    # already enough to flip the argmax to any non-degenerate alternative,
    # and two repeats (1.0) exceed the maximum possible score -- this is what
    # breaks the short bigram cycles ("of the most of the most...") since
    # those are caused by the same context always resolving to the same word.
    CONTEXT_REPEAT_PENALTY = 0.5

    # Smaller, rarity-weighted penalty for a word reappearing anywhere in the
    # walk so far. Common function words (high commonality) get a penalty near
    # zero -- repeating "the"/"of"/"and" is normal language -- while rare
    # content words are progressively discouraged from reappearing.
    GLOBAL_REPEAT_PENALTY = 0.12
    GLOBAL_REPEAT_CAP = 4

    # Number of extra "bark-similar" vocabulary words injected as candidates
    # at each step, so alpha has something to weigh even when the n-gram
    # table offers only a single real follower.
    WIDEN_TOP_K = 6

    def __init__(
        self,
        alpha: float = 0.5,
        max_words: int = 20,
        min_words: int = 5,
        spatial_weight: float = 0.35,
    ):
        """Initialize walk generator.

        Args:
            alpha: Blend factor. 0.0 = pure n-gram coherence, 1.0 = pure bark similarity.
            max_words: Maximum number of words in output.
            min_words: Don't stop at a sentence end before this many words, so the
                walk can't terminate after one or two tokens.
            spatial_weight: How much a per-step local patch of the bark's spatial
                grid (when supplied to generate()) blends into the steering
                vector, versus the rolled global feature vector. 0.0 = ignore
                the grid entirely, 1.0 = steer by the local patch alone.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0 inclusive")
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        if min_words < 1:
            raise ValueError("min_words must be >= 1")
        if not (0.0 <= spatial_weight <= 1.0):
            raise ValueError("spatial_weight must be between 0.0 and 1.0 inclusive")
        self.alpha = alpha
        self.max_words = max_words
        self.min_words = min_words
        self.spatial_weight = spatial_weight

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
    ) -> tuple[str, list[tuple[str, int]]] | None:
        """Pick the context key and candidate next words for the current position.

        Uses the two-word context ('prev current') when the corpus has a trigram
        table and that context is known; otherwise falls back to the bigram table
        for the current word. Returns None when neither offers a continuation.
        The returned context key is the exact lookup key used, so callers can
        track which (context, next-word) pairs have already been chosen.
        """
        current = words[-1]
        if corpus.trigram_table is not None and len(words) >= 2:
            key = f"{words[-2]} {current}"
            trigram = corpus.trigram_table.get(key)
            if trigram:
                return key, trigram
        bigram = corpus.bigram_table.get(current)
        if bigram:
            return current, bigram
        return None

    @staticmethod
    def _word_commonality(corpus: Corpus) -> dict[str, float]:
        """Rarity percentile per word (0=rarest, 1=most common), from bigram counts.

        Derived on the fly from the corpus's bigram table so no changes to the
        corpus/.npz format are needed. Used to scale the global repetition
        penalty: common function words should be repeatable, rare content
        words should not.
        """
        counts: dict[str, int] = {}
        for word, nexts in corpus.bigram_table.items():
            total = sum(count for _, count in nexts)
            counts[word] = counts.get(word, 0) + total
            for next_word, count in nexts:
                counts[next_word] = counts.get(next_word, 0) + count

        if not counts:
            return {}

        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        n = len(ranked)
        denom = max(n - 1, 1)
        return {word: i / denom for i, (word, _) in enumerate(ranked)}

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

    def _widen_candidates(
        self,
        candidates: list[tuple[str, int]],
        current_word: str,
        rolled: np.ndarray,
        corpus: Corpus,
    ) -> list[tuple[str, float]]:
        """Blend real n-gram candidates with synthetic bark-similar candidates.

        Real candidates keep their observed transition probability. Extra
        vocabulary words -- the top WIDEN_TOP_K by cosine similarity to the
        rolled bark vector, excluding words already in play -- are added with
        a low floor probability so they only win once alpha weights bark
        similarity heavily enough. This gives alpha real leverage in corpora
        where a word has only one observed follower (its own transition_prob
        of 1.0 would otherwise always win by default, regardless of alpha).
        """
        total_count = sum(count for _, count in candidates)
        scored: list[tuple[str, float]] = [
            (word, count / total_count) for word, count in candidates
        ]

        existing = {word for word, _ in candidates}
        existing.add(current_word)

        sims = self._cosine_similarities(rolled, corpus.word_embeddings)
        order = np.argsort(-sims, kind="stable")

        floor_prob = 1.0 / (total_count + self.WIDEN_TOP_K)
        added = 0
        for idx in order:
            if added >= self.WIDEN_TOP_K:
                break
            word = corpus.vocabulary[int(idx)]
            if word in existing:
                continue
            scored.append((word, floor_prob))
            existing.add(word)
            added += 1

        return scored

    def _score_candidates(
        self,
        candidates: list[tuple[str, float]],
        feature_vector: np.ndarray,
        corpus: Corpus,
        vocab_index: dict[str, int],
        context_key: str,
        context_choices: dict[str, dict[str, int]],
        word_counts: dict[str, int],
        commonality: dict[str, float],
    ) -> str:
        """Score candidates by blending transition probability, bark similarity,
        and repetition penalties, then return the argmax word.

        Args:
            candidates: List of (next_word, transition_probability) tuples
            feature_vector: Current (possibly rolled/steered) feature vector
            corpus: The corpus
            vocab_index: word -> index into corpus.vocabulary/word_embeddings
            context_key: The exact lookup key used to find these candidates,
                for tracking which (context, next-word) pairs recur
            context_choices: context_key -> {word: times chosen from this context}
            word_counts: word -> times it has appeared in the walk so far
            commonality: word -> rarity percentile (0=rarest, 1=most common)

        Returns:
            Selected next word
        """
        prior_context = context_choices.get(context_key, {})

        best_score = -float("inf")
        best_word = candidates[0][0]

        for word, transition_prob in candidates:
            idx = vocab_index.get(word)
            if idx is not None:
                sim = self._cosine_similarities(
                    feature_vector, corpus.word_embeddings[idx : idx + 1]
                )[0]
                bark_sim = (sim + 1) / 2  # Normalize [-1,1] to [0,1]
            else:
                bark_sim = 0.0

            context_penalty = self.CONTEXT_REPEAT_PENALTY * prior_context.get(word, 0)
            global_repeats = min(word_counts.get(word, 0), self.GLOBAL_REPEAT_CAP)
            global_penalty = (
                self.GLOBAL_REPEAT_PENALTY
                * (1.0 - commonality.get(word, 0.5))
                * global_repeats
            )

            score = (
                (1 - self.alpha) * transition_prob
                + self.alpha * bark_sim
                - context_penalty
                - global_penalty
            )
            if score > best_score:
                best_score = score
                best_word = word

        return best_word

    def _steering_vector(
        self,
        feature_vector: np.ndarray,
        step: int,
        stride: int,
        spatial_grid_norm: np.ndarray | None,
        step_cells: list[int] | None,
    ) -> np.ndarray:
        """Build the per-step vector used to compare against word embeddings.

        Always includes the rolled global feature vector (the pre-existing
        mechanism). When a normalized spatial grid is supplied, blends in the
        local descriptor for this step's grid cell -- tiled to match the
        embedding dimension -- so a specific patch of the bark image steers
        each word, not just an arbitrary reindexing of the whole vector.
        """
        rolled = np.roll(feature_vector, step * stride)
        if spatial_grid_norm is None or step_cells is None:
            return rolled

        cell = spatial_grid_norm[step_cells[step]]
        dim = feature_vector.shape[0]
        reps = -(-dim // cell.shape[0])  # ceil division
        tiled_cell = np.tile(cell, reps)[:dim]

        return (1 - self.spatial_weight) * rolled + self.spatial_weight * tiled_cell

    @staticmethod
    def _serpentine_path(grid_h: int, grid_w: int) -> list[int]:
        """Deterministic boustrophedon (serpentine) path over a grid_h x grid_w grid.

        Row 0 left-to-right, row 1 right-to-left, and so on, so consecutive
        steps stay spatially adjacent (no long jump from the end of one row to
        the start of the next), unlike a plain row-major raster scan.
        """
        path = []
        for i in range(grid_h):
            cols = range(grid_w) if i % 2 == 0 else range(grid_w - 1, -1, -1)
            path.extend(i * grid_w + j for j in cols)
        return path

    def _step_cells(self, n_cells: int) -> list[int]:
        """Map each walk step to a grid cell index via a subsampled serpentine path.

        Subsampled to exactly max_words points so a full poem always spans the
        entire grid, regardless of poem length versus the fixed cell count.
        """
        side = int(round(n_cells**0.5))
        full_path = self._serpentine_path(side, side)
        positions = np.linspace(0, len(full_path) - 1, num=self.max_words)
        indices = np.round(positions).astype(int)
        return [full_path[i] for i in indices]

    @staticmethod
    def _normalize_spatial_grid(spatial_grid: np.ndarray) -> np.ndarray:
        """Per-column standardize the spatial grid and clip to [-1, 1].

        Matches the scale convention of the image feature vector so the two
        can be blended meaningfully.
        """
        mean = spatial_grid.mean(axis=0)
        std = spatial_grid.std(axis=0)
        normalized = (spatial_grid - mean) / (std + 1e-8)
        return np.clip(normalized, -1, 1)

    def generate(
        self,
        feature_vector: np.ndarray,
        corpus: Corpus,
        spatial_grid: np.ndarray | None = None,
    ) -> str:
        """Generate text by walking the bigram model steered by the feature vector.

        Args:
            feature_vector: (D,) image feature vector
            corpus: Corpus with vocabulary, embeddings, bigram table, start words
            spatial_grid: Optional (N, K) per-cell local image descriptors (e.g.
                from ImageFeatureExtractor.extract_spatial_grid()). When given,
                each step's steering vector blends in the local descriptor for
                a specific grid cell, visited in a fixed reading order, so the
                bark's spatial structure -- not just its aggregate feature
                vector -- influences the walk. When omitted, behavior is
                identical to the vector-only steering used previously.

        Returns:
            Generated text string
        """
        stride = max(1, len(feature_vector) // self.max_words)
        vocab_index = {w: i for i, w in enumerate(corpus.vocabulary)}
        commonality = self._word_commonality(corpus)

        spatial_grid_norm = None
        step_cells = None
        if spatial_grid is not None:
            spatial_grid_norm = self._normalize_spatial_grid(spatial_grid)
            step_cells = self._step_cells(spatial_grid.shape[0])

        # Step 1: Pick first word from start_words, steered by step-0 vector.
        start_vector = self._steering_vector(
            feature_vector, 0, stride, spatial_grid_norm, step_cells
        )

        start_indices = []
        for word in corpus.start_words:
            idx = vocab_index.get(word)
            if idx is not None:
                start_indices.append((word, idx))

        if not start_indices:
            # Fallback: pick from entire vocabulary
            sims = self._cosine_similarities(start_vector, corpus.word_embeddings)
            best_idx = int(np.argmax(sims))
            current_word = corpus.vocabulary[best_idx]
        else:
            start_embeddings = np.array(
                [corpus.word_embeddings[idx] for _, idx in start_indices]
            )
            sims = self._cosine_similarities(start_vector, start_embeddings)
            best = int(np.argmax(sims))
            current_word = start_indices[best][0]

        words = [current_word]
        word_counts = {current_word: 1}
        context_choices: dict[str, dict[str, int]] = {}

        # Step 2: Walk
        for step in range(1, self.max_words):
            rolled = self._steering_vector(
                feature_vector, step, stride, spatial_grid_norm, step_cells
            )

            result = self._next_candidates(words, corpus)

            if result is not None:
                context_key, raw_candidates = result
                widened = self._widen_candidates(
                    raw_candidates, words[-1], rolled, corpus
                )
                current_word = self._score_candidates(
                    widened,
                    rolled,
                    corpus,
                    vocab_index,
                    context_key,
                    context_choices,
                    word_counts,
                    commonality,
                )
                context_choices.setdefault(context_key, {})
                context_choices[context_key][current_word] = (
                    context_choices[context_key].get(current_word, 0) + 1
                )
            else:
                # Dead-end fallback: pick from entire vocabulary by bark
                # similarity, penalized the same way for global repetition.
                sims = self._cosine_similarities(rolled, corpus.word_embeddings)
                bark = (sims + 1) / 2
                penalty = np.array(
                    [
                        self.GLOBAL_REPEAT_PENALTY
                        * (1.0 - commonality.get(w, 0.5))
                        * min(word_counts.get(w, 0), self.GLOBAL_REPEAT_CAP)
                        for w in corpus.vocabulary
                    ]
                )
                best_idx = int(np.argmax(bark - penalty))
                current_word = corpus.vocabulary[best_idx]

            words.append(current_word)
            word_counts[current_word] = word_counts.get(current_word, 0) + 1

            # Stop at a natural sentence end once we have enough words.
            if len(words) >= self.min_words and self._is_sentence_end(current_word):
                break

        # Post-processing: capitalize first character
        text = " ".join(words)
        text = text[0].upper() + text[1:] if text else ""

        return text
