# Gathering more Graz text

A practical to-do list for growing the `graz` corpus beyond the Wikipedia seed.
The loop is always the same:

> **Save text as `corpus_sources/graz/raw/<name>.txt` → run
> `bash corpus_sources/graz/build_graz.sh` → skim `cleaned/graz_all.txt`.**

`clean_text.py` repairs OCR and drops junk lines, but always eyeball the cleaned
file before trusting a noisy source. Aim to grow the vocabulary toward
~25k words (the builder prints it); more text = a livelier murmur.

Use UTF-8 plain text. One source per file makes it easy to remove a source later.

---

## 1. Historical Grazer newspapers — ANNO (the goldmine)

**What:** Full-text OCR of historical Austrian papers, public domain. Real
period commentary on the city. Titles: *Grazer Tagblatt, Grazer Volksblatt,
Tagespost (Graz), Grazer Zeitung, Grazer Mittags-Zeitung*.

**Where:** https://anno.onb.ac.at  → "Zeitungen & Zeitschriften" → search the title.

**How:**
1. Open an issue → click a page → the **OCR text** tab shows the recognized text.
2. Copy text from pages you like into `raw/anno_<title>_<year>.txt`.
3. Fraktur OCR is rough — that's fine, the cleaner handles the common cases, but
   delete obviously garbled pages.

**Tip:** pick a handful of full issues across a few decades rather than scattered
snippets; longer continuous text gives better bigram transitions.

---

## 2. Graz city-council protocols — Gemeinderat (best "commentary" register)

**What:** Minutes of city-council meetings. Literally people arguing about the
city. Official works → free to use (§7 UrhG).

**Where:** https://www.graz.at → search "Gemeinderat Protokolle" (Sitzungs­protokolle).

**How:**
1. Download the PDFs.
2. Convert to text: `pdftotext -layout protokoll.pdf raw/gemeinderat_2024.txt`
   (`pdftotext` ships with poppler-utils; `sudo apt install poppler-utils`).
3. These PDFs have headers/footers/speaker labels — the cleaner drops most, but
   check the result.

---

## 3. Stadt Graz press releases & open data

**What:** Modern civic German, clean text (no OCR).
**Where:** https://www.graz.at (Presseaussendungen) and https://data.graz.gv.at.
**How:** copy article text into `raw/graz_presse.txt`. For open-data CSVs with
text fields, export the relevant column to a `.txt`.

---

## 4. Landtag Steiermark protocols

**What:** Regional parliament stenographic protocols — formal civic German.
**Where:** https://www.landtag.steiermark.at → Protokolle.
**How:** same as Gemeinderat — download PDF → `pdftotext` → `raw/`.

---

## 5. Digitized public-domain books about Graz/Styria

**What:** Old Styriaca — chronicles, topographies, local histories.
**Where:**
- **ÖNB Digital / Austrian Books Online** — https://digital.onb.ac.at
- **Uni Graz GAMS / unipub** — https://gams.uni-graz.at
- **Steiermärkische Landesbibliothek** digital collections
- **Internet Archive** — https://archive.org/search?query=Graz (filter pre-1920)
- **Wikisource (de)** — https://de.wikisource.org
- Journals: *Mitteilungen des Historischen Vereines für Steiermark*,
  *Beiträge zur Kunde steiermärkischer Geschichtsquellen* (PD volumes on
  Archive/ANNO).

**How:** prefer the "full text" / `.txt` download where offered; otherwise
`pdftotext`. Drop into `raw/buch_<title>.txt`.

---

## 6. More Wikipedia (already scripted)

Add specific articles without editing code:

```bash
python corpus_sources/graz/fetch_wikipedia.py -o corpus_sources/graz/raw/wikipedia_extra.txt \
  --extra "Grazer Uhrturm" "Schloss Eggenberg" "Mariahilferkirche (Graz)" "Grazer Messe"
```

---

## Legal note

- **Official works** (council/parliament protocols, laws): free to use in Austria.
- **Historical works** (author died 70+ years ago): public domain.
- **Wikipedia:** CC BY-SA — keep the attribution already in the corpus `--source`.
- **Modern newspapers/opinion (Kleine Zeitung, ORF, etc.):** still in copyright.
  We deliberately left these out. The app is private/login-only, so quoting for a
  personal art piece is defensible, but don't redistribute the corpus with them in.

---

## When you've added a few sources

Re-run the build and re-check the numbers against the baselines:

```bash
bash corpus_sources/graz/build_graz.sh
./.venv/bin/python -m barkprints --list-corpora
```

Healthy targets (see the main README's built-in corpora table): vocabulary in
the 15k–25k range, and a "branching" share (words with >1 possible next word)
climbing toward `walden`'s ~42%. The higher the branching, the more the bark
`alpha` knob actually changes the output.
