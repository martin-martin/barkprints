#!/usr/bin/env bash
# Build the Graz corpus end-to-end from collected raw sources.
#
#   1. (optional) fetch the Wikipedia seed
#   2. clean every raw/*.txt into one corpus-ready file
#   3. build src/barkprints/corpora/graz.npz with the multilingual model
#
# Run from the repo root:  bash corpus_sources/graz/build_graz.sh
set -euo pipefail

HERE="corpus_sources/graz"
CLEANED="$HERE/cleaned/graz_all.txt"

# 1. Seed with Wikipedia if no raw text has been collected yet.
shopt -s nullglob
raw_files=("$HERE"/raw/*.txt)
if [ ${#raw_files[@]} -eq 0 ]; then
  echo ">> No raw/*.txt found — fetching the Wikipedia seed..."
  uv run python "$HERE/fetch_wikipedia.py" -o "$HERE/raw/wikipedia.txt"
  raw_files=("$HERE"/raw/*.txt)
fi

# 2. Clean all raw sources into one file.
echo ">> Cleaning ${#raw_files[@]} raw file(s)..."
uv run python "$HERE/clean_text.py" "${raw_files[@]}" -o "$CLEANED" --stats

# 3. Build the corpus.
echo ">> Building corpus..."
uv run python -m barkprints.corpus_builder \
  "$CLEANED" \
  src/barkprints/corpora/graz.npz \
  --name graz \
  --theme "Graz: Stadt, Chronik und Stimmen" \
  --source "Wikipedia (CC BY-SA); Gemeinderat Graz; ANNO/ÖNB & GAMS Uni Graz (PD)" \
  --model paraphrase-multilingual-MiniLM-L12-v2

echo ">> Done. Try:  uv run barkprints <bark>.jpg -c graz --alpha 0.5 --max-words 24"
