"""FastAPI app exposing barkprints as a mobile-friendly web service."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..corpus_loader import CorpusLoader
from ..text_generator import TextGenerator

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Barkprints", docs_url=None, redoc_url=None)

    generator = TextGenerator()
    loader = CorpusLoader()

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
        top_k: int = Form(1),
    ) -> JSONResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {image.content_type}",
            )
        if top_k < 1 or top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

        data = await image.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")

        suffix = Path(image.filename or "upload").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                result = generator.generate(tmp.name, corpus, top_k)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        if top_k == 1:
            return JSONResponse({"corpus": corpus, "matches": [{"text": result, "score": None}]})
        return JSONResponse(
            {
                "corpus": corpus,
                "matches": [
                    {"text": sentence, "score": float(score)} for sentence, score in result
                ],
            }
        )

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
