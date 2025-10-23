# Corpus Creation Guide 📚

This guide explains how to create custom text corpora for use with Barkprints.

## What is a Corpus?

A corpus in Barkprints is a collection of sentences with their pre-computed text embeddings. When you provide a bark image, the system finds which sentence's embedding is most similar to the image's feature vector.

## Quick Start

```bash
# 1. Create a text file
echo "Your sentences go here." > mycorpus.txt
echo "One sentence per line works best." >> mycorpus.txt

# 2. Build the corpus
uv run python -m barkprints.corpus_builder mycorpus.txt src/barkprints/corpora/mycorpus.npz

# 3. Use it
uv run barkprints barks.jpg -c mycorpus
```

## Corpus Builder Options

```bash
python -m barkprints.corpus_builder INPUT OUTPUT [OPTIONS]

Required:
  INPUT               Input text file
  OUTPUT              Output .npz corpus file

Options:
  --name NAME         Corpus name (default: input filename)
  --theme THEME       Theme/topic description
  --source SOURCE     Source attribution
  --model MODEL       Sentence-transformer model (default: all-MiniLM-L6-v2)
```

## Text Preparation

### Format

Your input text can be:
- One sentence per line
- Regular paragraphs (automatically split)
- Mixed format

The builder automatically:
- Splits on `.`, `!`, `?` followed by whitespace
- Filters out sentences shorter than 10 characters
- Cleans up whitespace

### Example Input File

```text
The ancient forest remembers every season.
Wisdom grows slowly like old-growth trees.
Each ring tells a story of survival.

Storms come and go but roots hold firm.
Patience is learned from watching seeds grow.
The forest teaches without speaking words.
```

## Best Practices

### Sentence Quality

✅ **Good sentences:**
- "The river flows endlessly toward the sea."
- "Time moves differently under the canopy."
- "Every fallen leaf returns to nourish the soil."

❌ **Avoid:**
- Too short: "Nice."
- Fragments: "Because of the rain"
- Lists: "apples, oranges, bananas"

### Corpus Size

- **Minimum**: 20-30 sentences for meaningful diversity
- **Optimal**: 50-100 sentences for good coverage
- **Maximum**: 1000+ sentences work but may be slow

Larger corpora provide more nuance but take longer to search.

### Thematic Coherence

Keep sentences thematically related:

✅ **Good theme cohesion:**
- All nature-related
- All philosophical quotes
- All news headlines
- All poetry lines

❌ **Avoid mixing:**
- Random Wikipedia facts
- Unrelated topics
- Different writing styles
- Mixed languages

### Content Diversity

Within your theme, include varied language:

```text
# Nature corpus - diverse perspectives
Ancient trees stand as silent witnesses.
Rain nourishes the awakening earth.
Branches reach skyward seeking light.
Decay creates fertile ground for new life.
The forest breathes with quiet wisdom.
```

This gives richer matching possibilities.

## Example Corpora

### Personal Journal

Extract meaningful sentences from your journal:

```bash
cat journal_2024.txt | grep -v "^$" > journal_corpus.txt
uv run python -m barkprints.corpus_builder journal_corpus.txt src/barkprints/corpora/journal.npz --theme "Personal reflections"
```

### Book Excerpts

Collect your favorite quotes from a book:

```text
Call me Ishmael.
It was the best of times, it was the worst of times.
All happy families are alike; each unhappy family is unhappy in its own way.
The only way out is through.
```

### Poetry Collection

Use lines from public domain poetry:

```bash
curl https://www.gutenberg.org/files/1337/1337-0.txt > poetry.txt
# Clean and select relevant lines
uv run python -m barkprints.corpus_builder poetry.txt src/barkprints/corpora/poetry.npz --theme "Classical poetry" --source "Project Gutenberg"
```

### News Headlines

Current events corpus:

```text
Scientists discover breakthrough in renewable energy.
Global cooperation needed to address climate crisis.
Technology continues to reshape modern life.
Cultural exchange bridges understanding between nations.
```

### Philosophical Quotes

Wisdom from philosophers:

```text
Know thyself and you will know the universe.
The unexamined life is not worth living.
We are what we repeatedly do.
The journey of a thousand miles begins with one step.
```

## Advanced: Custom Models

Use different sentence-transformer models for different languages or domains:

### Multilingual

```bash
# For non-English text
python -m barkprints.corpus_builder spanish.txt corpus.npz --model "paraphrase-multilingual-MiniLM-L12-v2"
```

### Larger Models (Better Quality)

```bash
# More powerful model (768 dimensions instead of 384)
python -m barkprints.corpus_builder text.txt corpus.npz --model "all-mpnet-base-v2"
```

Note: All corpora for a given project should use the same model for consistent embedding dimensions.

## Corpus File Format

The `.npz` format contains:

```python
{
    'sentences': np.array(['sentence 1', 'sentence 2', ...]),  # Text strings
    'embeddings': np.array([[...], [...], ...]),               # (N, D) embeddings
    'metadata': {                                               # Optional metadata
        'name': 'corpus_name',
        'theme': 'description',
        'source': 'where it came from',
        'model': 384,  # embedding dimension
        'num_sentences': 50
    }
}
```

You can also create these programmatically:

```python
from barkprints.corpus_builder import CorpusBuilder
import numpy as np

builder = CorpusBuilder(model_name='all-MiniLM-L6-v2')

text = """
Your text here.
Multiple sentences.
"""

sentences, embeddings, metadata = builder.build_from_text(
    text,
    corpus_name="mycorpus",
    metadata={"theme": "Custom theme"}
)

# Save
np.savez_compressed(
    'src/barkprints/corpora/mycorpus.npz',
    sentences=np.array(sentences, dtype=object),
    embeddings=embeddings,
    metadata=np.array(metadata)
)
```

## Performance Considerations

### Embedding Time

Building a corpus requires computing embeddings:
- ~0.1 seconds per sentence on CPU
- ~0.01 seconds per sentence on GPU
- 100 sentences ≈ 10 seconds total

### Search Time

Finding nearest match is very fast:
- Linear search: ~1ms for 100 sentences
- ~10ms for 1000 sentences
- No index needed for most use cases

### File Size

- Sentences: ~50-100 bytes each
- Embeddings: 384 floats × 4 bytes = 1.5KB each
- 100 sentences ≈ 150-200KB compressed

## Troubleshooting

### "No sentences extracted"

Your text might not have proper sentence boundaries. Try:
- Adding periods after each sentence
- Checking for encoding issues
- Manually splitting into sentences

### "Embeddings dimension mismatch"

All corpora must use same embedding dimension. Either:
- Rebuild all corpora with same model
- Keep separate projects for different models

### "Poor matches"

If matches seem unrelated:
- Increase corpus size (more options)
- Improve sentence quality
- Ensure thematic coherence
- Try different sentence-transformer model

## Getting More Text

### Public Domain Sources

- [Project Gutenberg](https://www.gutenberg.org/) - Classic literature
- [Wikiquote](https://en.wikiquote.org/) - Notable quotes
- [Poetry Foundation](https://www.poetryfoundation.org/) - Public domain poetry

### Generating Text

Use AI to generate thematic sentences:

```
"Generate 50 sentences about trees and nature, each 10-20 words, 
 philosophical and poetic in tone."
```

### Web Scraping

Respect robots.txt and terms of service:

```bash
# Example: Extract sentences from website
curl https://example.com/nature-blog | \
  sed 's/\. /.\n/g' | \
  grep -E '^[A-Z].*\.$' > corpus.txt
```

## Examples by Use Case

### Personal Meditation App

Collect calm, mindful sentences for peaceful bark readings.

### Educational Tool

Use sentences from textbooks in your subject area.

### Creative Writing

Generate story prompts from bark patterns.

### Philosophical Exploration

Deep questions and contemplations mapped to natural patterns.

### Cultural Archive

Preserve wisdom from oral traditions or cultural texts.

## Community Corpora

Share your corpora! Create a `corpora/` directory in your fork and submit pull requests with interesting text collections (respecting copyrights and attribution).

## Questions?

Open an issue on GitHub or check the main README for more examples.

