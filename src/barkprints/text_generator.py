"""Generate text from images using vocabulary and output formats."""

from random import Random

from .feature_extractor import ImageFeatureExtractor
from .format_loader import FormatLoader
from .vocabulary import Vocabulary
from .vocabulary_loader import VocabularyLoader


class TextGenerator:
    """Generate deterministic text from images."""
    
    def __init__(self):
        """Initialize text generator with loaders."""
        self.vocabulary_loader = VocabularyLoader()
        self.format_loader = FormatLoader()
    
    def generate(
        self,
        image_path: str,
        vocabulary_name: str,
        format_name: str | None = None
    ) -> str:
        """Generate text from an image using specified vocabulary.
        
        Args:
            image_path: Path to the image file
            vocabulary_name: Name of vocabulary to use
            format_name: Optional format override (uses vocabulary default if None)
            
        Returns:
            Generated text string
        """
        # Extract features and get seed
        extractor = ImageFeatureExtractor(image_path)
        seed = extractor.get_deterministic_seed()
        
        # Initialize seeded random generator
        rng = Random(seed)
        
        # Load vocabulary
        vocabulary = self.vocabulary_loader.load(vocabulary_name)
        
        # Determine format to use
        if format_name is None:
            format_name = vocabulary.output_format
        
        # Get format instance
        output_format = self.format_loader.get(format_name)
        
        # Generate text
        return output_format.generate(vocabulary, rng)

