"""Command-line interface for barkprints."""

import argparse
import sys
from pathlib import Path

from .corpus_loader import CorpusLoader
from .text_generator import TextGenerator


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate text from tree bark images using corpus-steered bigram walks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  barkprints barks.jpg -c nature
  barkprints barks.jpg -c nature --alpha 0.2
  barkprints barks.jpg -c nature --alpha 0.8 --max-words 30
  barkprints tree1.jpg tree2.jpg -c nature
  barkprints --list-corpora
        """,
    )

    parser.add_argument("images", nargs="*", help="Path(s) to bark image file(s)")

    parser.add_argument(
        "-c",
        "--corpus",
        default="nature",
        help="Corpus to use for text generation (default: nature)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend factor: 0.0 = bigram coherence, 1.0 = bark personality (default: 0.5)",
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=20,
        help="Maximum words in generated output (default: 20)",
    )

    parser.add_argument(
        "--list-corpora", action="store_true", help="List available corpora and exit"
    )

    args = parser.parse_args()

    # Handle list command
    if args.list_corpora:
        loader = CorpusLoader()
        corpora = loader.list_available()
        if corpora:
            print("Available corpora:")
            for corpus_name in corpora:
                try:
                    corpus = loader.load(corpus_name)
                    theme = corpus.metadata.get("theme", "unknown")
                    print(f"  - {corpus_name}: {len(corpus)} words ({theme})")
                except Exception as e:
                    print(f"  - {corpus_name}: Error loading - {e}")
        else:
            print("No corpora found.")
            print(
                "Create a corpus with: python -m barkprints.corpus_builder input.txt output.npz"
            )
        sys.exit(0)

    # Check if images were provided
    if not args.images:
        parser.error("the following arguments are required: images")
        sys.exit(1)

    # Generate text for each image
    generator = TextGenerator(alpha=args.alpha, max_words=args.max_words)
    error_occurred = False

    for image_path in args.images:
        path = Path(image_path)

        if not path.exists():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            error_occurred = True
            continue

        try:
            result = generator.generate(str(path), args.corpus)

            if len(args.images) > 1:
                print(f"\n{path.name}:")
                print("-" * 60)

            print(result)

            if len(args.images) > 1:
                print()

        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            error_occurred = True

    if error_occurred:
        sys.exit(1)


if __name__ == "__main__":
    main()
