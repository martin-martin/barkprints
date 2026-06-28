# Graz corpus sources

Working area for assembling the `graz` corpus — a large, German, Graz-themed
body of text so the trees of barkprints murmur in a recognizably-Graz voice.
Only the built `src/barkprints/corpora/graz.npz` ships in the app; nothing here
is needed at runtime.

## Pipeline

```
raw/*.txt  ──clean_text.py──▶  cleaned/graz_all.txt  ──corpus_builder──▶  graz.npz
```

One command does all of it (fetches the Wikipedia seed if `raw/` is empty):

```bash
bash corpus_sources/graz/build_graz.sh
```

## How "good" is measured here

The generator is an order-1 bigram walk, so quality = **corpus size + register +
language**, not the embedding model. Aim for a **vocabulary of ~10k–25k words**
(the builder prints it) so common words have many possible successors — that's
what gives the murmur variety. The tiny built-in corpora (154–235 words) are
~84% single-path and feel repetitive; `walden` (23k) and `rilke` (12k) are the
"good" baseline to beat.

## Sources (open / public-domain / official only)

Add each source as its own `raw/<name>.txt`, then re-run the build. Record
provenance in the `--source` metadata (already set in `build_graz.sh`).

### Automatable (already scripted)
- **German Wikipedia** — Graz, its 17 Bezirke, landmarks/institutions.
  `python fetch_wikipedia.py -o raw/wikipedia.txt` (CC BY-SA → attribute).
  Extend with `--extra "Title A" "Title B"`.

### Civic / contemporary (download → drop into raw/)
- **Graz Gemeinderat protocols** (city-council minutes) — graz.at. The best
  register fit: literally people commenting on the city. Export PDF → text.
- **Stadt Graz press releases & open data** — graz.at, data.graz.gv.at.
- **Landtag Steiermark** stenographic protocols — landtag.steiermark.at.

### Older digitized books & publications (public domain)
- **ANNO – AustriaN Newspapers Online** (anno.onb.ac.at) — full-text OCR of
  historical Grazer papers (Grazer Tagblatt, Grazer Volksblatt, Tagespost,
  Grazer Zeitung). The historical-commentary goldmine. Copy the OCR text.
- **ÖNB Digital / Austrian Books Online** (digital.onb.ac.at) — PD Styriaca.
- **Uni Graz GAMS / unipub** (gams.uni-graz.at) — digitized Styriaca.
- **Steiermärkische Landesbibliothek**, **Landesarchiv Steiermark**,
  **Universalmuseum Joanneum** publications.
- **Mitteilungen des Historischen Vereines für Steiermark** &
  **Beiträge zur Kunde steiermärkischer Geschichtsquellen** — PD journal
  volumes (Internet Archive / ANNO).
- **Internet Archive / Google Books / Wikisource (de)** — search "Graz" pre-1920.

### OCR caveat
Historical Fraktur OCR is noisy (long-s `ſ`, ligatures, hyphen-split words, page
furniture). `clean_text.py` repairs the common cases and drops gibberish lines,
but skim `cleaned/graz_all.txt` before building — bad tokens fragment the bigram
graph. Run with `--stats` to see how much survived.

## Deploy

`graz.npz` is auto-discovered by the loader once it's in
`src/barkprints/corpora/`. To ship it:

```bash
cd /home/martin/barkprints && docker compose up -d --build
```
