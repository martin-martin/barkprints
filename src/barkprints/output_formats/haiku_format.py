"""Haiku format implementation with 5-7-5 syllable structure."""

from random import Random

import pyphen

from ..vocabulary import Vocabulary
from .base_format import BaseOutputFormat


class HaikuFormat(BaseOutputFormat):
    """Generate haikus with 5-7-5 syllable structure."""
    
    def __init__(self):
        """Initialize haiku generator with syllable counter."""
        self.syllable_counter = pyphen.Pyphen(lang='en')
    
    @property
    def format_name(self) -> str:
        return "haiku"
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word.
        
        Args:
            word: Word to count syllables for
            
        Returns:
            Number of syllables
        """
        # Remove punctuation for counting
        clean_word = word.strip('.,!?;:').lower()
        hyphenated = self.syllable_counter.inserted(clean_word)
        return hyphenated.count('-') + 1
    
    def generate_line(
        self,
        target_syllables: int,
        word_pool: dict[str, list[str]],
        rng: Random
    ) -> str:
        """Generate a single haiku line with target syllable count.
        
        Args:
            target_syllables: Target number of syllables for the line
            word_pool: Dictionary mapping syllable counts to word lists
            rng: Random number generator
            
        Returns:
            Generated line as string
        """
        words = []
        current_syllables = 0
        
        # Try to build line with exact syllable count
        attempts = 0
        max_attempts = 100
        
        while current_syllables < target_syllables and attempts < max_attempts:
            attempts += 1
            needed = target_syllables - current_syllables
            
            # Find available word lengths that fit
            available_lengths = [
                length for length in word_pool.keys()
                if int(length) <= needed
            ]
            
            if not available_lengths:
                # Reset and try again
                words = []
                current_syllables = 0
                continue
            
            # Pick a syllable count and word
            syllable_key = rng.choice(available_lengths)
            word = rng.choice(word_pool[syllable_key])
            
            words.append(word)
            current_syllables += int(syllable_key)
        
        return ' '.join(words)
    
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        """Generate a haiku from the vocabulary.
        
        Args:
            vocabulary: Vocabulary with syllable-organized words
            rng: Seeded random number generator
            
        Returns:
            Three-line haiku string
            
        Raises:
            ValueError: If vocabulary is not syllable-organized
        """
        word_pool = vocabulary.words
        
        # Validate vocabulary structure
        if not all(key.isdigit() for key in word_pool.keys()):
            raise ValueError(
                f"Haiku format requires syllable-organized vocabulary. "
                f"Vocabulary '{vocabulary.name}' uses format '{vocabulary.output_format}'"
            )
        
        # Generate three lines: 5-7-5
        line1 = self.generate_line(5, word_pool, rng)
        line2 = self.generate_line(7, word_pool, rng)
        line3 = self.generate_line(5, word_pool, rng)
        
        return f"{line1}\n{line2}\n{line3}"

