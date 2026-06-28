#!/usr/bin/env python3
"""Clean raw German source text into a corpus-ready file for Barkprints.

Designed for the Graz corpus, where sources range from clean modern civic text
(Gemeinderat protocols, Wikipedia) to noisy historical OCR (ANNO/ÖNB Fraktur
scans). The bigram walk engine is only as good as its input: stray OCR tokens
and hyphen-split words fragment the bigram graph, so we repair and filter here.

Usage:
    python clean_text.py raw/*.txt -o cleaned/graz_all.txt
    cat raw/foo.txt | python clean_text.py - -o cleaned/foo.txt

The output is one sentence per line, lightly normalized, de-duplicated.
Run with --stats to see how much was dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

# --- Fraktur / OCR repair -------------------------------------------------

# Long-s and common ligatures that OCR leaves in historical scans.
_CHAR_FIXES = {
    "ſ": "s",  # ſ  long s
    "ꝛ": "r",  # ꝛ  r rotunda
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬆ": "st",
    "—": "-",  # em dash
    "–": "-",  # en dash
    "„": '"',  # „
    "“": '"',  # “
    "”": '"',  # ”
    "’": "'",
    "‘": "'",
    "­": "",   # soft hyphen
}

# A line is considered gibberish if too few of its tokens look like real words.
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _apply_char_fixes(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    return text


def _dehyphenate(text: str) -> str:
    """Join words split across a line break: 'Stadt-\\nplatz' -> 'Stadtplatz'."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _looks_like_prose(line: str, min_words: int = 4, min_word_ratio: float = 0.6) -> bool:
    """Heuristic: keep lines that read like real sentences, drop OCR noise."""
    tokens = line.split()
    if len(tokens) < min_words:
        return False
    wordish = sum(1 for t in tokens if _WORD_RE.search(t) and len(_WORD_RE.search(t).group()) >= 2)
    if wordish / len(tokens) < min_word_ratio:
        return False
    # Drop lines that are mostly digits/punctuation (page numbers, tables, refs).
    letters = sum(c.isalpha() for c in line)
    if letters < 0.5 * len(line):
        return False
    return True


def clean(text: str) -> list[str]:
    text = _apply_char_fixes(text)
    text = _dehyphenate(text)
    # Normalize whitespace but keep paragraph structure as spaces for splitting.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Flatten remaining single newlines inside paragraphs into spaces so the
    # sentence splitter sees continuous prose.
    text = re.sub(r"(?<![.!?])\n", " ", text)

    sentences = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    seen: set[str] = set()
    for s in sentences:
        s = s.strip()
        if len(s) <= 10:
            continue
        if not _looks_like_prose(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Input text files ('-' for stdin)")
    ap.add_argument("-o", "--output", required=True, help="Output .txt (one sentence per line)")
    ap.add_argument("--stats", action="store_true", help="Print drop statistics")
    args = ap.parse_args()

    raw_parts: list[str] = []
    for inp in args.inputs:
        if inp == "-":
            raw_parts.append(sys.stdin.read())
        else:
            raw_parts.append(Path(inp).read_text(encoding="utf-8", errors="replace"))
    raw = "\n".join(raw_parts)

    sentences = clean(raw)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

    if args.stats:
        words = sum(len(s.split()) for s in sentences)
        print(f"input chars : {len(raw):,}", file=sys.stderr)
        print(f"sentences   : {len(sentences):,}", file=sys.stderr)
        print(f"total words : {words:,}", file=sys.stderr)
    print(f"Wrote {len(sentences):,} sentences to {out_path}")


if __name__ == "__main__":
    main()
