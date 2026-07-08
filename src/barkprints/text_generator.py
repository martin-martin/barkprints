"""Generate text from images using corpus-steered n-gram walks."""

import numpy as np

from .corpus_loader import CorpusLoader
from .feature_extractor import ImageFeatureExtractor
from .walk_generator import WalkGenerator


def build_steering_features(
    extractor: ImageFeatureExtractor, target_dim: int, max_words: int
) -> np.ndarray:
    """Assemble the walk's steering matrix from a bark image.

    Row 0 is the whole-image feature vector (the tree's overall voice picks
    the start word); the remaining rows scan a window across the trunk left
    to right, one per walk step, so the poem is a transect of the bark.

    Returns:
        (max_words, target_dim) array
    """
    start_vector = extractor.extract_features(target_dim=target_dim)
    if max_words == 1:
        return start_vector[np.newaxis, :]
    transect = extractor.extract_transect_features(max_words - 1, target_dim=target_dim)
    return np.vstack([start_vector[np.newaxis, :], transect])


class TextGenerator:
    """Generate deterministic text from images via corpus walk."""

    def __init__(self, alpha: float = 0.5, max_words: int = 20, min_words: int = 5):
        """Initialize text generator.

        Args:
            alpha: Walk blend factor (0.0 = n-gram coherence, 1.0 = bark personality)
            max_words: Maximum words in generated output
            min_words: Minimum words before the walk may stop at a sentence end
        """
        self.corpus_loader = CorpusLoader()
        self.walk_generator = WalkGenerator(
            alpha=alpha, max_words=max_words, min_words=min_words
        )

    def generate(self, image_path: str, corpus_name: str) -> str:
        """Generate text from an image using specified corpus.

        Args:
            image_path: Path to the image file
            corpus_name: Name of corpus to use

        Returns:
            Generated text string
        """
        corpus = self.corpus_loader.load(corpus_name)
        embedding_dim = corpus.word_embeddings.shape[1]

        extractor = ImageFeatureExtractor(image_path)
        features = build_steering_features(
            extractor, embedding_dim, self.walk_generator.max_words
        )

        return self.walk_generator.generate(features, corpus)
