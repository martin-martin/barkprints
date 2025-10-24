# Barkprints 🌳

Generate text from tree bark images using embedding-based corpus matching.

## Concept

Barkprints explores finding hidden language in tree bark patterns. Instead of attempting to recognize what's in an image, it treats the numerical pixel data like it's already a text embedding and finds the most similar actual text from a corpus.

Every bark image produces a deterministic voice. Each unique bark pattern maps to a consistent position in embedding space, which matches to specific sentences in a chosen corpus. The same image always finds the same text, as if each tree speaks through its texture.

## How It Works

1. **Feature Extraction**: Extracts 384-dimensional feature vector from bark image
   - Color histograms, texture gradients, spatial statistics
   - Frequency features via DCT coefficients
   - Normalized to embedding-like range `[-1, 1]`

2. **Corpus Loading**: Loads pre-embedded text corpus
   - Sentences are embedded using `sentence-transformers`
   - Stored with their embeddings for fast matching

3. **Similarity Matching**: Finds nearest text via cosine similarity
   - Treats image features as if they were text embeddings
   - Computes similarity with all corpus embeddings
   - Returns the closest match(es)

4. **Deterministic Output**: Same image + corpus = same text, always

## Installation

Using uv (recommended):

```bash
cd barkprints
uv sync
```

## Quick Start

Generate text from a bark image:

```bash
barkprints barks.jpg -c nature
```

Try a different corpus:

```bash
barkprints barks.jpg -c literature
```

Get top 3 matches with similarity scores:

```bash
barkprints barks.jpg -c nature --top-k 3
```

Process multiple images:

```bash
barkprints tree1.jpg tree2.jpg tree3.jpg -c nature
```

## Usage

```bash
barkprints <image> [options]

Options:
  -c, --corpus NAME    Corpus to use (default: nature)
  --top-k K           Return top K matches (default: 1)
  --list-corpora      Show available corpora
```

## Built-in Corpora

### Nature

- **Theme**: Nature and forest wisdom
- **Size**: 50 sentences
- **Content**: Reflections on trees, seasons, growth, and natural cycles

### Literature

- **Theme**: Philosophical and literary quotes
- **Size**: 30 sentences
- **Content**: Classical wisdom and philosophical observations

## Creating Custom Corpora

Create a corpus from any text file:

```bash
python -m barkprints.corpus_builder input.txt output.npz --name mycorpus --theme "Your theme"
```

### Example: Create a News Corpus

```bash
# 1. Create a text file with sentences (one per line or paragraph)
cat > news.txt << EOF
Technology continues to reshape modern society.
Climate change demands urgent global action.
Scientific breakthroughs offer hope for the future.
EOF

# 2. Build the corpus with embeddings
uv run python -m barkprints.corpus_builder news.txt src/barkprints/corpora/news.npz --theme "Current events"

# 3. Use it
barkprints barks.jpg -c news
```

### Corpus Guidelines

- **Text quality**: Use well-formed, complete sentences
- **Sentence length**: 10-200 characters work best
- **Diversity**: Include varied language for richer matching
- **Theme coherence**: Keep sentences thematically related
- **Size**: 30-100 sentences is a good range

## Example Outputs

With `barks.jpg`:

```bash
$ barkprints barks.jpg -c nature
Death feeds new life in endless succession.

$ barkprints barks.jpg -c literature
The journey matters more than the destination itself.

$ barkprints barks.jpg -c nature --top-k 3
Top 3 matches:
1. [0.067] Death feeds new life in endless succession.
2. [0.063] Decay transforms into fertile soil again.
3. [0.062] Connection requires opening the heart completely.
```

## Programmatic Usage

```python
from barkprints.text_generator import TextGenerator

generator = TextGenerator()

# Get single match
text = generator.generate("barks.jpg", "nature")
print(text)

# Get top 3 matches with scores
matches = generator.generate("barks.jpg", "nature", top_k=3)
for sentence, score in matches:
    print(f"[{score:.3f}] {sentence}")

# The same image always produces the same output
text2 = generator.generate("barks.jpg", "nature")
assert text == text2  # Always True!
```

## Technical Details

**Feature Vector**: Extracts ~700 numerical features from images, then pads/truncates to match corpus embedding dimensions (typically 384 for all-MiniLM-L6-v2).

**Embedding Model**: Uses `all-MiniLM-L6-v2` by default (384 dimensions). Can use any sentence-transformer model by specifying `--model` when building corpora.

**Corpus Format**: `.npz` files containing:
- `sentences`: array of text strings
- `embeddings`: `(N, D)` matrix of embeddings
- `metadata`: dict with corpus info

**Similarity Metric**: Cosine similarity between normalized vectors. Higher scores indicate better matches.

**Determinism**: Same pixels → same features → same nearest neighbor → same text output

## Development

### Running Tests

```bash
uv run pytest
```

### Project Structure

```
barkprints/
├── src/barkprints/
│   ├── feature_extractor.py   # Image → 384-D feature vector
│   ├── corpus.py               # Corpus data structure
│   ├── corpus_loader.py        # Load .npz corpus files
│   ├── corpus_builder.py       # Build corpora from text
│   ├── embedding_matcher.py    # Cosine similarity matching
│   ├── text_generator.py       # Main generation pipeline
│   └── corpora/               # Built-in corpus files
│       ├── nature.npz
│       └── literature.npz
├── tests/                      # Test suite
├── pyproject.toml             # Python project config
└── README.md                   # This file
```

## Philosophy

This project treats images and text as inhabitants of the same conceptual space, a space of meaning represented numerically. By pretending that image features are text embeddings, we create a poetic bridge between visual texture and language. Each tree's unique bark pattern, their _barkprint_, becomes a coordinate in this shared space that points to a specific human expression.

For me, the project encourages me to look at trees and wonder what they may say, which opens a playful additional interaction layer with nature around me. It also addresses the interesting assumption that reality, human language, images of nature, could (can?) be compressed to a numerial representation. If everything is represented as numbers and we strip it down to that layer, how universal are these numbers? In this example of mixing image vectors and text vectors, I'm skipping a necessary translation step (one that works amazingly well these days, with CLIP etc.) by treating an image vector as a text vector. This idea makes me think about that there are different dialects of numeric languages that need translation between each other. There is a visual numeric dialect and a text-based dialect. Both are expressed in numbers, but the numbers mean different things. So in that sense, I think of it similarly to different languages. Anyways, just for fun, and climbing.

## License

MIT

## Credits

Created as an artistic exploration of the relationship between visual patterns and language through numerical representation.

Dependencies:
- [sentence-transformers](https://www.sbert.net/) for text embeddings
- [PIL/Pillow](https://python-pillow.org/) for image processing
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) for numerical operations
