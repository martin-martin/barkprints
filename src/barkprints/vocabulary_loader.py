"""Load and validate vocabulary sets from JSON files."""

import json
from pathlib import Path

from .vocabulary import Vocabulary


class VocabularyLoader:
    """Load vocabulary sets from JSON files."""
    
    def __init__(self, vocabularies_dir: Path | None = None):
        """Initialize loader with vocabularies directory.
        
        Args:
            vocabularies_dir: Directory containing vocabulary JSON files.
                            Defaults to package's vocabularies directory.
        """
        if vocabularies_dir is None:
            vocabularies_dir = Path(__file__).parent / "vocabularies"
        self.vocabularies_dir = Path(vocabularies_dir)
    
    def load(self, vocabulary_name: str) -> Vocabulary:
        """Load a vocabulary by name.
        
        Args:
            vocabulary_name: Name of vocabulary file (without .json extension)
            
        Returns:
            Loaded Vocabulary object
            
        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            ValueError: If vocabulary file is invalid
        """
        vocab_path = self.vocabularies_dir / f"{vocabulary_name}.json"
        
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary '{vocabulary_name}' not found at {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ['name', 'theme', 'output_format', 'words']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Vocabulary file missing required field: {field}")
        
        return Vocabulary(
            name=data['name'],
            theme=data['theme'],
            output_format=data['output_format'],
            words=data['words'],
            metadata=data.get('metadata', {})
        )
    
    def list_available(self) -> list[str]:
        """List all available vocabulary names.
        
        Returns:
            List of vocabulary names (without .json extension)
        """
        if not self.vocabularies_dir.exists():
            return []
        
        return [
            path.stem for path in self.vocabularies_dir.glob("*.json")
        ]

