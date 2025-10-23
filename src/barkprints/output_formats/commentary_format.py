"""Commentary format for generating natural sentences and paragraphs."""

from random import Random

from ..vocabulary import Vocabulary
from .base_format import BaseOutputFormat


class CommentaryFormat(BaseOutputFormat):
    """Generate commentary-style text with natural sentences."""
    
    @property
    def format_name(self) -> str:
        return "commentary"
    
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        """Generate commentary text from vocabulary.
        
        Args:
            vocabulary: Vocabulary with word lists
            rng: Seeded random number generator
            
        Returns:
            Generated commentary text
        """
        words = vocabulary.words
        
        # Support both flat word lists and categorized words
        if isinstance(words, dict) and 'subjects' in words:
            # Categorized format
            subject = rng.choice(words.get('subjects', ['something']))
            verb = rng.choice(words.get('verbs', ['is']))
            descriptor = rng.choice(words.get('descriptors', ['interesting']))
            context = rng.choice(words.get('context', ['today']))
            
            # Generate a simple sentence structure
            templates = [
                f"{subject.capitalize()} {verb} {descriptor} {context}.",
                f"{context.capitalize()}, {subject} {verb} {descriptor}.",
                f"The {subject} {verb} {descriptor}, {context}.",
            ]
            return rng.choice(templates)
        else:
            # Flat list format - pick random words
            all_words = []
            if isinstance(words, dict):
                for word_list in words.values():
                    if isinstance(word_list, list):
                        all_words.extend(word_list)
            elif isinstance(words, list):
                all_words = words
            
            if not all_words:
                return "No words available."
            
            # Generate a sentence with 5-8 words
            num_words = rng.randint(5, 8)
            selected_words = [rng.choice(all_words) for _ in range(num_words)]
            
            # Capitalize first word and add period
            sentence = ' '.join(selected_words)
            return sentence[0].upper() + sentence[1:] + '.'

