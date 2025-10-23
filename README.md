# Barkprints 🌳

Generate poetry and text from tree bark images using deterministic feature mapping.

## Concept

Barkprints is a creative project that explores the idea of finding "hidden language" in tree bark patterns. Rather than using computer vision to recognize what's in an image, it treats the numerical pixel data as a unique fingerprint that maps deterministically to words and text.

Every bark image produces a consistent "voice" - the same image always generates the same text. Different bark patterns create different poems or commentary, as if each tree is speaking through its unique texture.

## Features

- **Deterministic Generation**: Same image always produces the same output
- **Pluggable Vocabularies**: Load different word sets for different themes
- **Pluggable Output Formats**: Haiku, commentary, sentences, or custom formats
- **Modern Python**: Built with Python 3.10+, type hints, and UV package management
- **PyPI-Ready**: Structured for easy publishing as a Python package

## Installation

Using UV (recommended):

```bash
cd barkprints
uv sync
```

Or install directly:

```bash
uv pip install -e .
```

## Quick Start

Generate a haiku from a bark image:

```bash
barkprints barks.jpg -v nature
```

Generate news commentary:

```bash
barkprints barks.jpg -v news -f commentary
```

Process multiple images:

```bash
barkprints tree1.jpg tree2.jpg tree3.jpg -v nature
```

## Usage

```bash
barkprints <image> [options]

Options:
  -v, --vocabulary NAME    Vocabulary set to use (default: nature)
  -f, --format FORMAT      Output format override (default: vocabulary default)
  --list-vocabularies      Show available vocabularies
  --list-formats          Show available output formats
```

## Built-in Vocabularies

### Nature (default)
- **Theme**: Nature and seasons
- **Format**: Haiku (5-7-5 syllables)
- **Example Output**:
  ```
  ancient bark whispers
  golden autumn glistening
  shadow dance tranquil
  ```

### News
- **Theme**: Current events and commentary
- **Format**: Commentary sentences
- **Example Output**:
  ```
  The world evolves dramatically today.
  ```

## Creating Custom Vocabularies

Create a JSON file in `src/barkprints/vocabularies/`:

```json
{
  "name": "custom",
  "theme": "Your theme description",
  "output_format": "haiku",
  "words": {
    "1": ["word", "list", "here"],
    "2": ["longer", "words", "here"],
    "3": ["syllables", "matter", "for haiku"]
  },
  "metadata": {
    "description": "Custom vocabulary",
    "version": "1.0"
  }
}
```

For non-haiku formats, organize words by category:

```json
{
  "name": "custom",
  "theme": "Your theme",
  "output_format": "commentary",
  "words": {
    "subjects": ["things", "people"],
    "verbs": ["do", "say"],
    "descriptors": ["quickly", "slowly"],
    "context": ["today", "now"]
  }
}
```

## Creating Custom Output Formats

1. Create a new file in `src/barkprints/output_formats/`:

```python
from random import Random
from ..vocabulary import Vocabulary
from .base_format import BaseOutputFormat

class MyFormat(BaseOutputFormat):
    @property
    def format_name(self) -> str:
        return "myformat"
    
    def generate(self, vocabulary: Vocabulary, rng: Random) -> str:
        # Your generation logic here
        words = vocabulary.words
        # Use rng for deterministic randomness
        return "Generated text"
```

2. Register it in `format_loader.py`:

```python
from .output_formats import MyFormat

# In FormatLoader.__init__:
self.register(MyFormat())
```

## How It Works

1. **Feature Extraction**: Extracts numerical features from the image (color histograms, texture gradients, statistical measures)
2. **Deterministic Seeding**: Hashes the feature vector to create a consistent seed
3. **Word Selection**: Uses the seed to deterministically select words from the vocabulary
4. **Text Generation**: Applies the output format's rules to construct the final text

## Development

This project uses UV for dependency management and modern Python practices:

- Python 3.10+ with modern type hints
- No `from __future__ import` needed
- Structured with `src/` layout for clean packaging
- Ready for PyPI publishing

### Running Tests

Install dev dependencies and run the test suite:

```bash
uv sync --all-groups
uv run pytest
```

Run tests with coverage report:

```bash
uv run pytest --cov=barkprints --cov-report=html
```

The test suite includes:
- **45 tests** covering all components
- **65% code coverage**
- Unit tests for feature extraction, vocabularies, and formats
- Integration tests for end-to-end workflows
- CLI tests for command-line interface
- Determinism tests to verify reproducibility

## License

MIT

## Credits

Created as an artistic exploration of finding meaning in nature's patterns.

