"""Build corpora from text with word-level embeddings and bigram tables."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import Corpus


class CorpusBuilder:
    """Build word-level corpus files from text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize corpus builder.

        Args:
            model_name: Name of sentence-transformer model to use
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def tokenize(self, sentences: list[str]) -> list[list[str]]:
        """Split sentences into lowercase words, preserving attached punctuation.

        Args:
            sentences: List of sentences

        Returns:
            List of token lists
        """
        tokenized = []
        for sentence in sentences:
            words = sentence.lower().split()
            if words:
                tokenized.append(words)
        return tokenized

    def build_corpus(
        self,
        text: str,
        corpus_name: str,
        metadata: dict | None = None,
    ) -> Corpus:
        """Build a word-level corpus from text.

        Args:
            text: Input text (multiple sentences)
            corpus_name: Name for the corpus
            metadata: Optional metadata dict

        Returns:
            Corpus object with word-level data
        """
        # Split into sentences
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s) > 10]
        print(f"Extracted {len(sentences)} sentences")

        # Tokenize
        tokenized = self.tokenize(sentences)
        if not tokenized:
            raise ValueError(
                "No usable tokens found. Input text may be too short or all "
                "sentences were filtered by the length threshold."
            )

        # Build vocabulary (sorted for determinism)
        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        vocabulary = sorted(all_words)
        print(f"Vocabulary size: {len(vocabulary)} words")

        # Build bigram table
        bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tokens in tokenized:
            for i in range(len(tokens) - 1):
                bigram_counts[tokens[i]][tokens[i + 1]] += 1

        bigram_table: dict[str, list[tuple[str, int]]] = {}
        for word, nexts in bigram_counts.items():
            bigram_table[word] = [(next_word, count) for next_word, count in sorted(nexts.items())]

        # Identify start words (sorted for determinism)
        start_words = sorted(set(tokens[0] for tokens in tokenized if tokens))
        print(f"Start words: {len(start_words)}")

        # Embed each vocabulary word
        print("Generating word embeddings...")
        word_embeddings = self.model.encode(
            vocabulary, show_progress_bar=True, convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {word_embeddings.shape}")

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["name"] = corpus_name
        metadata["model"] = self.model.get_sentence_embedding_dimension()
        metadata["num_words"] = len(vocabulary)

        return Corpus(
            name=corpus_name,
            vocabulary=vocabulary,
            word_embeddings=word_embeddings,
            bigram_table=bigram_table,
            start_words=start_words,
            metadata=metadata,
        )

    def build_from_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        corpus_name: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Build corpus from text file and save to .npz.

        Args:
            input_path: Path to input text file
            output_path: Path to output .npz file
            corpus_name: Name for corpus (defaults to input filename)
            metadata: Optional metadata dict
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if corpus_name is None:
            corpus_name = input_path.stem

        print(f"Reading {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        corpus = self.build_corpus(text, corpus_name, metadata)
        save_corpus(corpus, output_path)


def save_corpus(corpus: Corpus, output_path: str | Path) -> None:
    """Save a Corpus object to .npz format.

    Args:
        corpus: Corpus to save
        output_path: Path to output .npz file
    """
    output_path = Path(output_path)

    # Serialize bigram table as JSON
    bigram_json = json.dumps(corpus.bigram_table)

    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        vocabulary=np.array(corpus.vocabulary, dtype=object),
        word_embeddings=corpus.word_embeddings,
        start_words=np.array(corpus.start_words, dtype=object),
        bigram_json=np.array(bigram_json),
        metadata=np.array(corpus.metadata),
    )
    print(f"Corpus saved: {len(corpus.vocabulary)} words, {corpus.word_embeddings.shape}")


def main():
    """CLI entry point for corpus building."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a word-level corpus from text file with embeddings"
    )
    parser.add_argument("input", help="Input text file")
    parser.add_argument("output", help="Output .npz corpus file")
    parser.add_argument("--name", help="Corpus name (default: input filename)")
    parser.add_argument("--theme", help="Theme/topic of corpus")
    parser.add_argument("--source", help="Source of text")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    metadata = {}
    if args.theme:
        metadata["theme"] = args.theme
    if args.source:
        metadata["source"] = args.source

    builder = CorpusBuilder(model_name=args.model)
    builder.build_from_file(
        args.input, args.output, corpus_name=args.name, metadata=metadata
    )


if __name__ == "__main__":
    main()
