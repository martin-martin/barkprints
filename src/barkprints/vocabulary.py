"""Vocabulary data structures and utilities."""

from typing import Any


class Vocabulary:
    """Represents a vocabulary set with metadata."""
    
    def __init__(
        self,
        name: str,
        theme: str,
        output_format: str,
        words: dict[str, Any],
        metadata: dict[str, Any] | None = None
    ):
        """Initialize vocabulary.
        
        Args:
            name: Name of the vocabulary
            theme: Theme description
            output_format: Default output format to use
            words: Word data structure (format-specific)
            metadata: Additional metadata
        """
        self.name = name
        self.theme = theme
        self.output_format = output_format
        self.words = words
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Vocabulary(name='{self.name}', theme='{self.theme}', format='{self.output_format}')"

