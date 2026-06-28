"""FastAPI app exposing barkprints as a mobile-friendly web service.

Beyond generating text from bark images, the app lets logged-in users *save*
an image together with its generated text, a timestamp, and (optionally) the
geolocation where the photo was taken, then browse those saved entries in a
per-user gallery. Auth is intentionally minimal: a small set of accounts
created out-of-band (see ``adduser.py``), bcrypt password hashing, and a signed
session cookie. No signup flow.
"""

from __future__ import annotations

import mimetypes
import os
import secrets
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..corpus import Corpus
from ..corpus_loader import CorpusLoader
from ..feature_extractor import ImageFeatureExtractor
from ..walk_generator import WalkGenerator
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

COOKIE_NAME = "bp_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
COOKIE_SECURE = os.environ.get("BARKPRINTS_COOKIE_SECURE", "").lower() in {"1", "true", "yes"}


def _data_dir() -> Path:
    return Path(os.environ.get("BARKPRINTS_DATA_DIR", "data")).resolve()


def _load_secret_key(data_dir: Path) -> str:
    """Resolve the cookie-signing key.

    Prefers ``BARKPRINTS_SECRET_KEY``; otherwise persists a random key under the
    data dir so sessions survive restarts without any required configuration.
    """
    env = os.environ.get("BARKPRINTS_SECRET_KEY")
    if env:
        return env
    key_file = data_dir / "secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_urlsafe(48)
    key_file.write_text(key)
    key_file.chmod(0o600)
    return key


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Barkprints", docs_url=None, redoc_url=None)

    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir)
    serializer = URLSafeTimedSerializer(_load_secret_key(data_dir), salt="bp-session")

    loader = CorpusLoader()

    @lru_cache(maxsize=None)
    def get_corpus(name: str) -> Corpus:
        # Corpora are immutable once built; cache them so large embedding
        # files (e.g. walden, ~34 MB) are read from disk only once per process.
        return loader.load(name)

    # -- auth helpers ----------------------------------------------------

    def current_user_id(request: Request) -> Optional[int]:
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return None
        try:
            data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        except (BadSignature, SignatureExpired):
            return None
        uid = data.get("uid") if isinstance(data, dict) else None
        if uid is None or store.get_user(uid) is None:
            return None
        return int(uid)

    def require_user(request: Request) -> int:
        uid = current_user_id(request)
        if uid is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return uid

    def set_session(response: Response, user_id: int) -> None:
        token = serializer.dumps({"uid": user_id})
        response.set_cookie(
            COOKIE_NAME, token,
            max_age=SESSION_MAX_AGE, httponly=True,
            samesite="lax", secure=COOKIE_SECURE, path="/",
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -- pages -----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index(request: Request) -> Response:
        if current_user_id(request) is None:
            return RedirectResponse("/login", status_code=303)
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request) -> Response:
        if current_user_id(request) is not None:
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC_DIR / "login.html")

    @app.get("/gallery", include_in_schema=False)
    def gallery_page(request: Request) -> Response:
        if current_user_id(request) is None:
            return RedirectResponse("/login", status_code=303)
        return FileResponse(STATIC_DIR / "gallery.html")

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

    # -- auth API --------------------------------------------------------

    @app.post("/api/login")
    async def api_login(username: str = Form(...), password: str = Form(...)) -> Response:
        user = store.verify_login(username, password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        response = JSONResponse({"username": user.username})
        set_session(response, user.id)
        return response

    @app.post("/api/logout")
    def api_logout() -> Response:
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/me")
    def api_me(request: Request) -> JSONResponse:
        uid = current_user_id(request)
        if uid is None:
            return JSONResponse({"user": None})
        user = store.get_user(uid)
        return JSONResponse({"user": user.username if user else None})

    # -- corpora ---------------------------------------------------------

    @app.get("/api/corpora")
    def list_corpora(user_id: int = Depends(require_user)) -> JSONResponse:
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

    # -- generation ------------------------------------------------------

    @app.post("/api/generate")
    async def generate(
        image: UploadFile = File(...),
        corpus: str = Form("nature"),
        alpha: float = Form(0.5),
        max_words: int = Form(20),
        user_id: int = Depends(require_user),
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

    # -- saved entries ---------------------------------------------------

    @app.post("/api/save")
    async def save_entry(
        image: UploadFile = File(...),
        text: str = Form(...),
        corpus: str = Form("nature"),
        alpha: Optional[float] = Form(None),
        max_words: Optional[int] = Form(None),
        lat: Optional[float] = Form(None),
        lon: Optional[float] = Form(None),
        accuracy: Optional[float] = Form(None),
        user_id: int = Depends(require_user),
    ) -> JSONResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type: {image.content_type}",
            )
        if not text.strip():
            raise HTTPException(status_code=400, detail="Cannot save empty text")

        data = await image.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")

        suffix = Path(image.filename or "upload").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
            suffix = ".jpg"
        filename = f"{uuid4().hex}{suffix}"
        (store.uploads_dir / filename).write_bytes(data)

        entry = store.add_entry(
            user_id=user_id, corpus=corpus, alpha=alpha, max_words=max_words,
            text=text, image_filename=filename, lat=lat, lon=lon, accuracy=accuracy,
        )
        return JSONResponse({"entry": entry.as_dict()}, status_code=201)

    @app.get("/api/entries")
    def list_entries(user_id: int = Depends(require_user)) -> JSONResponse:
        entries = store.list_entries(user_id)
        return JSONResponse({"entries": [e.as_dict() for e in entries]})

    @app.get("/api/entries/{entry_id}/image", include_in_schema=False)
    def entry_image(entry_id: int, user_id: int = Depends(require_user)) -> FileResponse:
        entry = store.get_entry(entry_id, user_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Not found")
        path = store.uploads_dir / entry.image_filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image missing")
        media_type = mimetypes.guess_type(entry.image_filename)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.delete("/api/entries/{entry_id}")
    def delete_entry(entry_id: int, user_id: int = Depends(require_user)) -> JSONResponse:
        filename = store.delete_entry(entry_id, user_id)
        if filename is None:
            raise HTTPException(status_code=404, detail="Not found")
        try:
            (store.uploads_dir / filename).unlink(missing_ok=True)
        except OSError:
            pass
        return JSONResponse({"ok": True})

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
