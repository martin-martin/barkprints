"""Corpus data structures for text embedding matching."""

import numpy as np


class Corpus:
    """Represents a text corpus with pre-computed embeddings."""
    
    def __init__(
        self,
        name: str,
        sentences: list[str],
        embeddings: np.ndarray,
        metadata: dict | None = None
    ):
        """Initialize corpus.
        
        Args:
            name: Name of the corpus
            sentences: List of text sentences
            embeddings: (N, 768) array of sentence embeddings
            metadata: Additional metadata (theme, source, etc.)
        """
        self.name = name
        self.sentences = sentences
        self.embeddings = embeddings
        self.metadata = metadata or {}
        
        # Validate
        if len(sentences) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(sentences)} sentences but {len(embeddings)} embeddings"
            )
        
        # Allow any embedding dimension (384, 768, etc.)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array, got shape {embeddings.shape}"
            )
    
    def __len__(self) -> int:
        """Return number of sentences in corpus."""
        return len(self.sentences)
    
    def __repr__(self) -> str:
        return f"Corpus(name='{self.name}', size={len(self)}, theme='{self.metadata.get('theme', 'unknown')}')"

