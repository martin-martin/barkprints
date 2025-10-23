"""Base class for output format plugins."""

from abc import ABC, abstractmethod
from random import Random

from ..vocabulary import Vocabulary


class BaseOutputFormat(ABC):
    """Abstract base class for output format implementations."""
    
    @abstractmethod
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        """Generate text using the vocabulary and random number generator.
        
        Args:
            vocabulary: Vocabulary object with words and metadata
            rng: Seeded Random instance for deterministic generation
            
        Returns:
            Generated text string
        """
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this format."""
        pass

