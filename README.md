# Barkprints 🌳

Generate text from tree bark images by walking a corpus, steered by the image's texture.

## Concept

Barkprints explores finding hidden language in tree bark patterns. Instead of recognizing what's in an image, it treats the numerical pixel data like a text embedding and lets it steer a walk through the words of a chosen corpus.

Every bark image produces a deterministic voice. Each unique bark pattern maps to a consistent position in feature space, which guides a word-by-word walk through a corpus. The same image always finds the same text, as if each tree speaks through its texture.

## How It Works

1. **Feature Extraction**: Extracts a 384-dimensional feature vector from the bark image
   - Color histograms, texture gradients, spatial statistics
   - Frequency features via DCT coefficients
   - Normalized to embedding-like range `[-1, 1]`
   - A separate 10×10 grid of local per-patch descriptors (brightness, contrast,
     edge density) is also extracted, for the spatial steering below.

2. **Corpus**: A corpus is built ahead of time from a body of text and stored as an `.npz` file containing:
   - the **vocabulary** (unique words)
   - a **word embedding** for each word (`sentence-transformers`)
   - a **bigram table** (which words follow which, with counts)
   - the set of **start words** (words that begin sentences)

3. **Steered Walk**: The generator walks the bigram/trigram model one word at a time.
   - The first word is chosen from the corpus's start words by bark similarity.
   - At each step the candidate next words (from the bigram/trigram table, widened with a few bark-similar
     vocabulary words so `alpha` always has real choices to weigh) are scored by blending:
     `score = (1 - alpha) * transition_probability + alpha * bark_similarity - repetition_penalty`
   - The repetition penalty discourages re-choosing a word from a context the walk has already been in, and
     (more gently) discourages a word from reappearing at all — without ever forbidding it outright, so
     function words can still repeat naturally while short cycles ("of the most of the most...") can't.
   - The image feature vector is *rolled* a little at each step, and blended with the local descriptor for a
     specific patch of the bark — visited in a fixed serpentine reading order across the image — so a
     specific part of the bark's structure steers each word, not just an arbitrary reindexing of the whole vector.

4. **Deterministic Output**: Same image + corpus + settings = same text, always. There is no randomness
   anywhere in the pipeline — every choice is a strict argmax over the scoring formula above.

### The `alpha` knob

`alpha` blends coherence against bark personality:

- `alpha = 0.0` — pure bigram/trigram coherence: follow the most common word transitions (reads smoothest, ignores the image).
- `alpha = 1.0` — pure bark personality: pick whichever next word the image is most "similar" to (wilder, more image-driven).
- `alpha = 0.5` (default) — a blend of the two.

Even where the n-gram table offers only one observed follower, a handful of bark-similar vocabulary words are
added as low-probability alternatives, so `alpha` has real leverage at every step — not just in the larger corpora.

### The `spatial-weight` knob

`spatial-weight` (`--spatial-weight`, default `0.35`) controls how much a specific patch of the bark image —
visited in a fixed reading order as the poem progresses — blends into the steering vector, versus the bark's
overall color/texture/frequency signature:

- `spatial-weight = 0.0` — steer by the whole image's aggregate feature vector only.
- `spatial-weight = 1.0` — steer by the local patch alone, so each word reflects one specific part of the bark.

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
cd barkprints
uv sync                 # core (CLI)
uv sync --extra web     # also install the web/PWA server
```

## Quick Start (CLI)

Generate text from a bark image:

```bash
uv run barkprints barks.jpg -c nature
```

Turn up the bark personality and length:

```bash
uv run barkprints barks.jpg -c walden --alpha 0.8 --max-words 30
```

Process multiple images:

```bash
uv run barkprints tree1.jpg tree2.jpg tree3.jpg -c walden
```

### CLI Options

```bash
barkprints <image>... [options]

  -c, --corpus NAME       Corpus to use (default: nature)
  --alpha FLOAT           Blend: 0.0 = bigram/trigram coherence, 1.0 = bark personality (default: 0.5)
  --max-words INT         Maximum words in output (default: 20)
  --min-words INT         Minimum words before the walk may stop at a sentence end (default: 5)
  --spatial-weight FLOAT  How much a local bark patch steers each step, vs. the whole image (default: 0.35)
  --list-corpora          Show available corpora and exit
```

## Web App / PWA

Barkprints ships with a mobile-friendly web interface (FastAPI + an installable PWA) so you can snap a bark photo on your phone and hear it speak.

```bash
uv sync --extra web
uv run barkprints-web                       # http://127.0.0.1:8000
uv run barkprints-web --host 0.0.0.0 --port 8000   # expose to your LAN / a reverse proxy
```

Open the URL on your phone and "Add to Home Screen" for an app-like experience. The UI lets you pick a corpus, set the **bark influence** (`alpha`), **max words**, and **bark structure** (`spatial-weight`), and take or upload a photo.

### Accounts & saving

The app is private: every page and API route (except `/login` and health) requires a
logged-in session. There is **no signup** — create accounts from the command line:

```bash
uv run python -m barkprints.web.adduser alice            # prompts for a password
uv run python -m barkprints.web.adduser alice --password s3cret   # non-interactive
uv run python -m barkprints.web.adduser alice --update   # change an existing password
uv run python -m barkprints.web.adduser --list           # list usernames
```

Once signed in, generating a poem reveals a **Save this** button and a small map. The
pin is seeded automatically — from the photo's own **GPS EXIF** tag if it has one
(extracted server-side; authoritative for a library photo taken elsewhere), otherwise
from your device's current location — and you can **drag it to the exact tree** or tap
the map to place it. Saving stores the photo, the generated text, a timestamp, and the
chosen **location** (so you can find the tree again). Saved entries appear in the
per-user **Saved** gallery, which plots every located entry on a map (markers open a
photo + text popup) above the list of cards. The map uses a **vendored copy of Leaflet**
(`static/vendor/leaflet/`, BSD-2); only the OpenStreetMap tile images are fetched at
runtime.

Persisted data (a SQLite DB, uploaded photos, and the cookie-signing key) lives under
`BARKPRINTS_DATA_DIR` (default `data/`). Auth uses bcrypt password hashes and a signed
session cookie; set `BARKPRINTS_SECRET_KEY` to keep sessions valid across restarts
(otherwise a random key is persisted in the data dir).

API endpoints: `GET /api/corpora`, `POST /api/generate` (multipart: `image`, `corpus`,
`alpha`, `max_words`, `spatial_weight`; the response includes `exif_lat`/`exif_lon` when
the photo carries GPS metadata), `POST /api/login` / `POST /api/logout` / `GET /api/me`,
`POST /api/save` (multipart: `image`, `text`, `corpus`, `alpha`, `max_words`,
`spatial_weight`, optional `lat`/`lon`/`accuracy`), `GET /api/entries`,
`GET /api/entries/{id}/image`, `DELETE /api/entries/{id}`, `GET /healthz`.

## Deployment (Docker)

Production runs at **https://barkprints.quest** on a VPS that also hosts another app
(`soundmap`). That app's Caddy container already owns ports 80/443 and handles TLS, so
barkprints does **not** run its own reverse proxy — it sits behind the existing Caddy:

- `Dockerfile` — FastAPI/uvicorn app. It installs only the runtime deps
  (`fastapi`, `uvicorn`, `python-multipart`, `pillow`, `numpy`, `scipy`, `bcrypt`,
  `itsdangerous`). `sentence-transformers`/`torch` are deliberately **excluded**: they're
  only needed to *build* corpora, never to serve requests.
- `docker-compose.yml` — runs the `barkprints` container, bound to `127.0.0.1:5051` for
  local checks, and attached to the **external** `soundmap_web` Docker network so Caddy
  can reach it as `http://barkprints:8000`. A named volume `barkprints_data` is mounted at
  `/app/data` to persist the SQLite DB, uploaded photos, and the signing key across
  rebuilds; `BARKPRINTS_COOKIE_SECURE=1` is set because the app is only reached over HTTPS.

Create accounts inside the running container (no signup flow exists):

```bash
docker compose exec barkprints python -m barkprints.web.adduser alice
```
- The site block (`barkprints.quest, www.barkprints.quest` → `reverse_proxy
  barkprints:8000`, with www→apex redirect) lives in the **soundmap project's
  `Caddyfile`**, not here.

Deploy / redeploy app code:

```bash
docker compose up -d --build
```

> ⚠️ **Caddyfile gotcha — reload won't pick up edits.** Caddy mounts the `Caddyfile` as a
> single file. Editing it on the host changes the file's inode, but the container's mount
> still points at the old one, so `caddy reload` silently keeps serving the **old** config
> (it logs `config is unchanged`). The symptom: a newly added site returns a TLS error
> (`tlsv1 alert internal error`) because Caddy never learned about it and so never got a
> cert. **Fix: restart the Caddy container** so the file is re-mounted:
>
> ```bash
> # run from the soundmap project dir (where the Caddyfile + caddy service live)
> docker compose restart caddy
> ```
>
> After the restart, the first HTTPS request triggers Let's Encrypt issuance and may fail
> for a few seconds before the cert is ready — retry and it'll come up.

## Built-in Corpora

| Corpus       | Words  | Theme                                   |
|--------------|--------|-----------------------------------------|
| `nature`     | 235    | Nature and forest wisdom (small)        |
| `literature` | 154    | Philosophical and literary quotes (small) |
| `walden`     | 23,502 | Thoreau — nature, wilderness, contemplation |
| `tao`        | 2,706  | The Way and virtue                      |
| `rilke`      | 12,572 | Rilke — Poesie, Existenz und Wahrnehmung (German) |

List them anytime:

```bash
uv run barkprints --list-corpora
```

## Creating Custom Corpora

Build a corpus from any text file (see [CORPUS_GUIDE.md](CORPUS_GUIDE.md) for details):

```bash
uv run python -m barkprints.corpus_builder input.txt src/barkprints/corpora/mycorpus.npz \
  --name mycorpus --theme "Your theme"
```

The builder splits the text into sentences, tokenizes them into words, learns the bigram transitions, and embeds each vocabulary word. Building requires `sentence-transformers` (installed via `uv sync`); **running** the generator does not.

## Example Outputs

With `barks.jpg`:

```bash
$ uv run barkprints barks.jpg -c nature
Creatures find shelter among twisted roots.

$ uv run barkprints barks.jpg -c walden --alpha 0.0 --max-words 16
Fishes, and the most of the same time to the woods and a few miles of

$ uv run barkprints barks.jpg -c walden --alpha 1.0 --max-words 16
Fishes, pressure _point feature. inclines, cloying prevalent mercy.
```

(Even at `alpha = 0.0`, the repetition penalty keeps the walk from looping on its single most-likely
path — it used to get stuck repeating "of the most" forever. Raising `alpha` lets the bark pull the walk
further from plain n-gram coherence.)

## Programmatic Usage

```python
from barkprints.text_generator import TextGenerator

generator = TextGenerator(alpha=0.8, max_words=30, spatial_weight=0.35)

text = generator.generate("barks.jpg", "walden")
print(text)

# The same image + corpus + settings always produces the same output
assert generator.generate("barks.jpg", "walden") == text  # always True
```

## Technical Details

**Feature Vector**: ~700 numerical features per image, normalized and padded/truncated to the corpus embedding dimension (384 for `all-MiniLM-L6-v2`). A separate 10×10 spatial grid of local per-patch descriptors (brightness, contrast, gradient magnitude, edge density) is extracted alongside it.

**Embedding Model**: Word embeddings are produced at *build* time with `all-MiniLM-L6-v2` (384 dimensions) by default; choose another with `--model` when building a corpus.

**Corpus Format** (`.npz`): `vocabulary`, `word_embeddings` `(V, D)`, `bigram_json` (serialized transition table), `trigram_json` (optional), `start_words`, and `metadata`.

**Similarity Metric**: Cosine similarity between the (rolled and spatially-steered) image vector and word embeddings.

**Anti-Repetition**: Candidates are penalized for having already been chosen from the same n-gram context, and (more gently) for having already appeared anywhere in the walk — enough to break short cycles without forbidding natural repetition of common words.

**Alpha-Widening**: A few extra vocabulary words, chosen by bark similarity, are added as low-probability candidates at every step, so `alpha` has real choices to weigh even where the n-gram table offers only one observed follower.

**Determinism**: Same pixels → same features → same scored walk → same text.

## Development

```bash
uv run pytest          # run the test suite
```

### Project Structure

```
barkprints/
├── src/barkprints/
│   ├── feature_extractor.py   # Image → 384-D feature vector
│   ├── corpus.py              # Corpus data structure (vocab, embeddings, bigrams)
│   ├── corpus_loader.py       # Load .npz corpus files
│   ├── corpus_builder.py      # Build corpora from text (needs sentence-transformers)
│   ├── walk_generator.py      # Bark-steered bigram walk
│   ├── text_generator.py      # Main generation pipeline
│   ├── web/                   # FastAPI + PWA web interface
│   └── corpora/               # Built-in corpus .npz files
├── tests/
├── Dockerfile                 # Runtime image (no torch/sentence-transformers)
├── docker-compose.yml         # Container, joins soundmap's Caddy network
├── pyproject.toml
└── README.md
```

## Philosophy

This project treats images and text as inhabitants of the same conceptual space, a space of meaning represented numerically. By pretending that image features are text embeddings, we create a poetic bridge between visual texture and language. Each tree's unique bark pattern, their _barkprint_, becomes a coordinate in this shared space that points to specific human expression.

For me, the project encourages me to look at trees and wonder what they may say, which opens a playful additional interaction layer with nature around me. It also addresses the interesting assumption that reality, human language, images of nature, could (can?) be compressed to a numerial representation. If everything is represented as numbers and we strip it down to that layer, how universal are these numbers? In this example of mixing image vectors and text vectors, I'm skipping a necessary translation step (one that works amazingly well these days, with CLIP etc.) by treating an image vector as a text vector. This idea makes me think about that there are different dialects of numeric languages that need translation between each other. There is a visual numeric dialect and a text-based dialect. Both are expressed in numbers, but the numbers mean different things. So in that sense, I think of it similarly to different languages. Anyways, just for fun, and climbing.

## License

MIT

## Credits

Created as an artistic exploration of the relationship between visual patterns and language through numerical representation.

Dependencies:
- [sentence-transformers](https://www.sbert.net/) for word embeddings (corpus building)
- [PIL/Pillow](https://python-pillow.org/) for image processing
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) for numerical operations
- [FastAPI](https://fastapi.tiangolo.com/) & [uvicorn](https://www.uvicorn.org/) for the web interface
</content>
</invoke>
