"""Command-line interface for barkprints."""

import argparse
import sys
from pathlib import Path

from .text_generator import TextGenerator
from .vocabulary_loader import VocabularyLoader
from .format_loader import FormatLoader


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate poetry and text from tree bark images using deterministic feature mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  barkprints barks.jpg -v nature
  barkprints barks.jpg -v news -f commentary
  barkprints tree1.jpg tree2.jpg -v nature
        """
    )
    
    parser.add_argument(
        'images',
        nargs='*',
        help='Path(s) to bark image file(s)'
    )
    
    parser.add_argument(
        '-v', '--vocabulary',
        default='nature',
        help='Vocabulary set to use (default: nature)'
    )
    
    parser.add_argument(
        '-f', '--format',
        default=None,
        help='Output format override (default: use vocabulary default)'
    )
    
    parser.add_argument(
        '--list-vocabularies',
        action='store_true',
        help='List available vocabularies and exit'
    )
    
    parser.add_argument(
        '--list-formats',
        action='store_true',
        help='List available output formats and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_vocabularies:
        loader = VocabularyLoader()
        vocabs = loader.list_available()
        print("Available vocabularies:")
        for vocab in vocabs:
            print(f"  - {vocab}")
        sys.exit(0)
    
    if args.list_formats:
        loader = FormatLoader()
        formats = loader.list_available()
        print("Available output formats:")
        for fmt in formats:
            print(f"  - {fmt}")
        sys.exit(0)
    
    # Check if images were provided
    if not args.images:
        parser.error("the following arguments are required: images")
        sys.exit(1)
    
    # Generate text for each image
    generator = TextGenerator()
    error_occurred = False
    
    for image_path in args.images:
        path = Path(image_path)
        
        if not path.exists():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            error_occurred = True
            continue
        
        try:
            text = generator.generate(
                str(path),
                args.vocabulary,
                args.format
            )
            
            # Print with nice formatting
            if len(args.images) > 1:
                print(f"\n{path.name}:")
                print("-" * 40)
            
            print(text)
            
            if len(args.images) > 1:
                print()
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            error_occurred = True
    
    if error_occurred:
        sys.exit(1)


if __name__ == '__main__':
    main()

