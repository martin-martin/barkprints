"""Simple sentence format for generating single sentences."""

from random import Random

from ..vocabulary import Vocabulary
from .base_format import BaseOutputFormat


class SentenceFormat(BaseOutputFormat):
    """Generate simple single sentences."""
    
    @property
    def format_name(self) -> str:
        return "sentence"
    
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        """Generate a single sentence from vocabulary.
        
        Args:
            vocabulary: Vocabulary with word lists
            rng: Seeded random number generator
            
        Returns:
            Generated sentence
        """
        words = vocabulary.words
        
        # Extract all words from vocabulary structure
        all_words = []
        if isinstance(words, dict):
            for word_list in words.values():
                if isinstance(word_list, list):
                    all_words.extend(word_list)
        elif isinstance(words, list):
            all_words = words
        
        if not all_words:
            return "No words available."
        
        # Generate a sentence with 4-6 words
        num_words = rng.randint(4, 6)
        selected_words = [rng.choice(all_words) for _ in range(num_words)]
        
        # Capitalize first word and add period
        sentence = ' '.join(selected_words)
        return sentence[0].upper() + sentence[1:] + '.'

