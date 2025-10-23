"""Load and manage output format plugins."""

from .output_formats import BaseOutputFormat, HaikuFormat, CommentaryFormat, SentenceFormat


class FormatLoader:
    """Registry for output format plugins."""
    
    def __init__(self):
        """Initialize format registry with built-in formats."""
        self._formats: dict[str, BaseOutputFormat] = {}
        
        # Register built-in formats
        self.register(HaikuFormat())
        self.register(CommentaryFormat())
        self.register(SentenceFormat())
    
    def register(self, format_instance: BaseOutputFormat) -> None:
        """Register a new output format.
        
        Args:
            format_instance: Instance of an output format class
        """
        self._formats[format_instance.format_name] = format_instance
    
    def get(self, format_name: str) -> BaseOutputFormat:
        """Get a format by name.
        
        Args:
            format_name: Name of the format
            
        Returns:
            Format instance
            
        Raises:
            KeyError: If format not found
        """
        if format_name not in self._formats:
            raise KeyError(f"Format '{format_name}' not found. Available: {self.list_available()}")
        return self._formats[format_name]
    
    def list_available(self) -> list[str]:
        """List all registered format names.
        
        Returns:
            List of format names
        """
        return list(self._formats.keys())

