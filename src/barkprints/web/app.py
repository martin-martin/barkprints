"""FastAPI app exposing barkprints as a mobile-friendly web service."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..corpus import Corpus
from ..corpus_loader import CorpusLoader
from ..feature_extractor import ImageFeatureExtractor
from ..walk_generator import WalkGenerator

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Barkprints", docs_url=None, redoc_url=None)

    loader = CorpusLoader()

    @lru_cache(maxsize=None)
    def get_corpus(name: str) -> Corpus:
        # Corpora are immutable once built; cache them so large embedding
        # files (e.g. walden, ~34 MB) are read from disk only once per process.
        return loader.load(name)

    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def manifest() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "manifest.webmanifest",
            media_type="application/manifest+json",
        )

    @app.get("/sw.js", include_in_schema=False)
    def service_worker() -> FileResponse:
        # Serve from the root scope so the SW can control the whole app.
        return FileResponse(
            STATIC_DIR / "sw.js",
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )

    @app.get("/api/corpora")
    def list_corpora() -> JSONResponse:
        names = loader.list_available()
        items = []
        for name in sorted(names):
            try:
                corpus = loader.load(name)
                items.append(
                    {
                        "name": name,
                        "size": len(corpus),
                        "theme": corpus.metadata.get("theme", ""),
                    }
                )
            except Exception:
                items.append({"name": name, "size": None, "theme": ""})
        return JSONResponse({"corpora": items})

    @app.post("/api/generate")
    async def generate(
        image: UploadFile = File(...),
        corpus: str = Form("nature"),
        alpha: float = Form(0.5),
        max_words: int = Form(20),
    ) -> JSONResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {image.content_type}",
            )
        if not (0.0 <= alpha <= 1.0):
            raise HTTPException(status_code=400, detail="alpha must be between 0.0 and 1.0")
        if not (1 <= max_words <= 200):
            raise HTTPException(status_code=400, detail="max_words must be between 1 and 200")

        data = await image.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")

        try:
            corpus_obj = get_corpus(corpus)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        suffix = Path(image.filename or "upload").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                extractor = ImageFeatureExtractor(tmp.name)
                features = extractor.extract_features(
                    target_dim=corpus_obj.word_embeddings.shape[1]
                )
                text = WalkGenerator(alpha=alpha, max_words=max_words).generate(
                    features, corpus_obj
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse({"corpus": corpus, "text": text})

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
