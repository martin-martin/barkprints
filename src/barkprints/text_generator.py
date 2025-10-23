"""Generate text from images using corpus embedding matching."""

from .corpus_loader import CorpusLoader
from .embedding_matcher import EmbeddingMatcher
from .feature_extractor import ImageFeatureExtractor


class TextGenerator:
    """Generate deterministic text from images via embedding matching."""
    
    def __init__(self):
        """Initialize text generator with loaders."""
        self.corpus_loader = CorpusLoader()
        self.matcher = EmbeddingMatcher()
    
    def generate(
        self,
        image_path: str,
        corpus_name: str,
        top_k: int = 1
    ) -> str | list[tuple[str, float]]:
        """Generate text from an image using specified corpus.
        
        Args:
            image_path: Path to the image file
            corpus_name: Name of corpus to use
            top_k: Number of top matches to return (1 = single text, >1 = list with scores)
            
        Returns:
            If top_k=1: Single matched sentence (str)
            If top_k>1: List of (sentence, similarity_score) tuples
        """
        # Load corpus first to get embedding dimension
        corpus = self.corpus_loader.load(corpus_name)
        embedding_dim = corpus.embeddings.shape[1]
        
        # Extract features matching corpus embedding dimension
        extractor = ImageFeatureExtractor(image_path)
        image_embedding = extractor.extract_features(target_dim=embedding_dim)
        
        # Find nearest sentence(s)
        matches = self.matcher.find_nearest(image_embedding, corpus, top_k)
        
        # Return format based on top_k
        if top_k == 1:
            return matches[0][0]  # Just the sentence
        else:
            return matches  # List of (sentence, score) tuples
