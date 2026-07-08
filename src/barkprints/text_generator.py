"""Generate text from images using corpus-steered bigram walks."""

from .corpus_loader import CorpusLoader
from .feature_extractor import ImageFeatureExtractor
from .walk_generator import WalkGenerator


class TextGenerator:
    """Generate deterministic text from images via bigram walk."""

    def __init__(
        self,
        alpha: float = 0.5,
        max_words: int = 20,
        min_words: int = 5,
        spatial_weight: float = 0.35,
    ):
        """Initialize text generator.

        Args:
            alpha: Walk blend factor (0.0 = n-gram coherence, 1.0 = bark personality)
            max_words: Maximum words in generated output
            min_words: Minimum words before the walk may stop at a sentence end
            spatial_weight: How much the bark's local spatial structure (versus
                its aggregate feature vector) steers each step of the walk
        """
        self.corpus_loader = CorpusLoader()
        self.walk_generator = WalkGenerator(
            alpha=alpha,
            max_words=max_words,
            min_words=min_words,
            spatial_weight=spatial_weight,
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
        feature_vector = extractor.extract_features(target_dim=embedding_dim)
        spatial_grid = extractor.extract_spatial_grid()

        return self.walk_generator.generate(feature_vector, corpus, spatial_grid=spatial_grid)
