"""Load and manage text corpora with embeddings."""

from pathlib import Path

import numpy as np

from .corpus import Corpus


class CorpusLoader:
    """Load corpora from .npz files."""
    
    def __init__(self, corpora_dir: Path | None = None):
        """Initialize loader with corpora directory.
        
        Args:
            corpora_dir: Directory containing corpus .npz files.
                        Defaults to package's corpora directory.
        """
        if corpora_dir is None:
            corpora_dir = Path(__file__).parent / "corpora"
        self.corpora_dir = Path(corpora_dir)
    
    def load(self, corpus_name: str) -> Corpus:
        """Load a corpus by name.
        
        Args:
            corpus_name: Name of corpus file (without .npz extension)
            
        Returns:
            Loaded Corpus object
            
        Raises:
            FileNotFoundError: If corpus file doesn't exist
            ValueError: If corpus file is invalid
        """
        corpus_path = self.corpora_dir / f"{corpus_name}.npz"
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus '{corpus_name}' not found at {corpus_path}")
        
        # Load npz file
        data = np.load(corpus_path, allow_pickle=True)
        
        # Validate required fields
        required_fields = ['sentences', 'embeddings']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Corpus file missing required field: {field}")
        
        # Extract data
        sentences = data['sentences'].tolist()
        embeddings = data['embeddings']
        metadata = data.get('metadata', np.array({}))[()]
        
        # Handle metadata conversion
        if not isinstance(metadata, dict):
            metadata = {}
        
        return Corpus(
            name=corpus_name,
            sentences=sentences,
            embeddings=embeddings,
            metadata=metadata
        )
    
    def list_available(self) -> list[str]:
        """List all available corpus names.
        
        Returns:
            List of corpus names (without .npz extension)
        """
        if not self.corpora_dir.exists():
            return []
        
        return [
            path.stem for path in self.corpora_dir.glob("*.npz")
        ]

