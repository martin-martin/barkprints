"""Match image feature vectors to text corpus using embedding similarity."""

import numpy as np

from .corpus import Corpus


class EmbeddingMatcher:
    """Match image embeddings to corpus sentences via cosine similarity."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def compute_similarities(
        image_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities between image and all corpus embeddings.
        
        Args:
            image_embedding: (D,) image feature vector
            corpus_embeddings: (N, D) corpus embedding matrix
            
        Returns:
            (N,) array of similarity scores
        """
        # Normalize vectors
        image_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-10)
        corpus_norms = corpus_embeddings / (
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        )
        
        # Compute dot products (cosine similarity for normalized vectors)
        similarities = np.dot(corpus_norms, image_norm)
        
        return similarities
    
    def find_nearest(
        self,
        image_embedding: np.ndarray,
        corpus: Corpus,
        top_k: int = 1
    ) -> list[tuple[str, float]]:
        """Find nearest sentences to image embedding.
        
        Args:
            image_embedding: (D,) image feature vector
            corpus: Corpus to search
            top_k: Number of top matches to return
            
        Returns:
            List of (sentence, similarity_score) tuples
        """
        # Compute similarities
        similarities = self.compute_similarities(image_embedding, corpus.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return sentences with scores
        results = [
            (corpus.sentences[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results

