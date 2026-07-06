"""Upgrade legacy .npz corpora to cleaned tokenization (one-time dev tool).

Legacy corpora tokenized text by whitespace, so 'fishes,' and 'fishes' were
distinct vocabulary entries with punctuation attached, and Gutenberg markup
like '_quercus' leaked into the vocabulary. This tool re-derives a cleaned
corpus from the legacy tables alone (no source text needed):

- every legacy token is cleaned with the current builder tokenization
  (punctuation split off; a token may clean into several words),
- bigram edges are remapped onto the cleaned words with counts summed,
- end-word statistics are estimated from the sentence-final punctuation of
  the legacy variants, weighted by how often each variant occurred,
- the cleaned vocabulary is re-embedded with the sentence-transformer model.

Requires the 'build' extra (sentence-transformers). Occurrence counts are
estimated from the bigram table (max of a token's incoming and outgoing
totals), which is exact for mid-sentence words and a safe floor elsewhere.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from .corpus import Corpus
from .corpus_builder import WORD_RE, save_corpus
from .corpus_loader import CorpusLoader
from .walk_generator import _TRAILING


def _clean(token: str) -> list[str]:
    """Clean a legacy token into zero or more bare lowercase words."""
    return WORD_RE.findall(token.lower())


def _ends_with_terminal_punctuation(token: str) -> bool:
    """Whether a legacy token carries sentence-final punctuation."""
    stripped = token.rstrip(_TRAILING)
    return bool(stripped) and stripped[-1] in ".!?"


def migrate_corpus(corpus: Corpus, model) -> Corpus:
    """Return a cleaned version of a legacy corpus.

    Args:
        corpus: Legacy corpus (punctuation attached to tokens)
        model: SentenceTransformer used to re-embed the cleaned vocabulary
    """
    if corpus.end_words is not None:
        raise ValueError(f"Corpus '{corpus.name}' already has cleaned tokenization")
    if corpus.trigram_table is not None:
        raise ValueError(
            f"Corpus '{corpus.name}' has a trigram table; rebuild it from its "
            "source text instead of migrating"
        )

    # Estimate each legacy token's occurrence count from the bigram table:
    # outgoing total (as a source) and incoming total (as a target). Sentence-
    # final tokens have no outgoing edges and sentence-initial ones may have
    # no incoming edges, so take the max, with a floor of one occurrence.
    outgoing: dict[str, int] = defaultdict(int)
    incoming: dict[str, int] = defaultdict(int)
    for source, followers in corpus.bigram_table.items():
        for target, count in followers:
            outgoing[source] += count
            incoming[target] += count

    def occurrences(token: str) -> int:
        return max(outgoing[token], incoming[token], 1)

    cleaned = {token: _clean(token) for token in corpus.vocabulary}

    # Remap bigram edges onto cleaned words. An edge a -> b becomes
    # last(clean(a)) -> first(clean(b)); a token that cleans into several
    # words additionally contributes its internal chain, weighted by its
    # occurrence estimate.
    bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source, followers in corpus.bigram_table.items():
        source_words = cleaned.get(source) or _clean(source)
        if not source_words:
            continue
        for target, count in followers:
            target_words = cleaned.get(target) or _clean(target)
            if not target_words:
                continue
            bigram_counts[source_words[-1]][target_words[0]] += count
    for token, words in cleaned.items():
        for first, second in zip(words, words[1:]):
            bigram_counts[first][second] += occurrences(token)

    bigram_table = {
        word: sorted(followers.items()) for word, followers in bigram_counts.items()
    }

    # End-word statistics from the legacy punctuation, occurrence-weighted.
    end_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    for token, words in cleaned.items():
        if not words:
            continue
        count = occurrences(token)
        for word in words:
            total_counts[word] += count
        if _ends_with_terminal_punctuation(token):
            end_counts[words[-1]] += count
    end_words = {
        word: (end_counts[word], total_counts[word]) for word in sorted(end_counts)
    }

    start_words = sorted(
        {words[0] for token in corpus.start_words if (words := _clean(token))}
    )

    vocabulary = sorted({word for words in cleaned.values() for word in words})
    print(
        f"{corpus.name}: {len(corpus.vocabulary)} legacy tokens -> "
        f"{len(vocabulary)} cleaned words, {len(end_words)} end words"
    )

    print("Re-embedding cleaned vocabulary...")
    word_embeddings = model.encode(
        vocabulary, show_progress_bar=True, convert_to_numpy=True
    )

    metadata = dict(corpus.metadata)
    metadata["num_words"] = len(vocabulary)
    metadata["cleaned"] = True

    return Corpus(
        name=corpus.name,
        vocabulary=vocabulary,
        word_embeddings=word_embeddings,
        bigram_table=bigram_table,
        start_words=start_words,
        metadata=metadata,
        end_words=end_words,
    )


def main() -> None:
    """CLI entry point: migrate named corpora in place."""
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(
        description="Upgrade legacy .npz corpora to cleaned tokenization (in place)"
    )
    parser.add_argument("names", nargs="+", help="Corpus names (without .npz)")
    parser.add_argument(
        "--corpora-dir",
        type=Path,
        default=None,
        help="Directory of corpus files (default: built-in corpora)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    loader = CorpusLoader(corpora_dir=args.corpora_dir)
    model = SentenceTransformer(args.model)

    for name in args.names:
        corpus = loader.load(name)
        migrated = migrate_corpus(corpus, model)
        save_corpus(migrated, loader.corpora_dir / f"{name}.npz")


if __name__ == "__main__":
    main()
