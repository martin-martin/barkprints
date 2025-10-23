"""Build corpora from text files with sentence embeddings."""

import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class CorpusBuilder:
    """Build corpus files from text with sentence embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize corpus builder.
        
        Args:
            model_name: Name of sentence-transformer model to use
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")
    
    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s) > 10]
        
        return sentences
    
    def build_from_text(
        self,
        text: str,
        corpus_name: str,
        metadata: dict | None = None
    ) -> tuple[list[str], np.ndarray, dict]:
        """Build corpus from text string.
        
        Args:
            text: Input text
            corpus_name: Name for the corpus
            metadata: Optional metadata dict
            
        Returns:
            Tuple of (sentences, embeddings, metadata)
        """
        # Split into sentences
        sentences = self.split_into_sentences(text)
        print(f"Extracted {len(sentences)} sentences")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata['name'] = corpus_name
        metadata['model'] = self.model.get_sentence_embedding_dimension()
        metadata['num_sentences'] = len(sentences)
        
        return sentences, embeddings, metadata
    
    def build_from_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        corpus_name: str | None = None,
        metadata: dict | None = None
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
        
        # Read text file
        print(f"Reading {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Build corpus
        sentences, embeddings, final_metadata = self.build_from_text(
            text, corpus_name, metadata
        )
        
        # Save to npz
        print(f"Saving to {output_path}...")
        np.savez_compressed(
            output_path,
            sentences=np.array(sentences, dtype=object),
            embeddings=embeddings,
            metadata=np.array(final_metadata)
        )
        print(f"Corpus saved: {len(sentences)} sentences, {embeddings.shape}")


def main():
    """CLI entry point for corpus building."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build a corpus from text file with embeddings"
    )
    parser.add_argument('input', help='Input text file')
    parser.add_argument('output', help='Output .npz corpus file')
    parser.add_argument('--name', help='Corpus name (default: input filename)')
    parser.add_argument('--theme', help='Theme/topic of corpus')
    parser.add_argument('--source', help='Source of text')
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Prepare metadata
    metadata = {}
    if args.theme:
        metadata['theme'] = args.theme
    if args.source:
        metadata['source'] = args.source
    
    # Build corpus
    builder = CorpusBuilder(model_name=args.model)
    builder.build_from_file(
        args.input,
        args.output,
        corpus_name=args.name,
        metadata=metadata
    )


if __name__ == '__main__':
    main()

