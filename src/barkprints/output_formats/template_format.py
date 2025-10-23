"""Template for creating custom output formats.

Copy this file, rename it, and implement your custom text generation logic.
Then register it in format_loader.py.
"""

from random import Random

from ..vocabulary import Vocabulary
from .base_format import BaseOutputFormat


class TemplateFormat(BaseOutputFormat):
    """Template for custom output format implementation."""
    
    @property
    def format_name(self) -> str:
        """Return a unique name for this format."""
        return "template"
    
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        """Generate text using the vocabulary and random number generator.
        
        Args:
            vocabulary: Vocabulary object with words and metadata
            rng: Seeded Random instance for deterministic generation
                 Use rng.choice(), rng.randint(), etc. for all randomness
            
        Returns:
            Generated text string
        """
        # Example: Access vocabulary words
        words = vocabulary.words
        
        # Example: Extract words based on your vocabulary structure
        # For syllable-based (like haiku):
        if isinstance(words, dict) and '1' in words:
            all_words = []
            for syllable_list in words.values():
                all_words.extend(syllable_list)
        # For category-based (like commentary):
        elif isinstance(words, dict) and 'subjects' in words:
            subjects = words.get('subjects', [])
            verbs = words.get('verbs', [])
            # ... use your categories
        
        # Example: Use rng for deterministic randomness
        # word = rng.choice(all_words)
        # num = rng.randint(1, 10)
        
        # Implement your text generation logic here
        return "Your generated text"


# To use this format:
# 1. Copy this file and rename it (e.g., my_format.py)
# 2. Rename the class (e.g., MyFormat)
# 3. Change format_name property to your format name
# 4. Implement your generate() logic
# 5. In format_loader.py, add:
#    from .output_formats import MyFormat
#    self.register(MyFormat())

