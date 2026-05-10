"""Entry point for ``barkprints-web``: serves the mobile-friendly UI."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Barkprints mobile-friendly web server.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("BARKPRINTS_HOST", "127.0.0.1"),
        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose to LAN/proxy).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BARKPRINTS_PORT", "8000")),
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (development only).",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing web dependencies. Install with: uv sync --extra web"
        ) from exc

    uvicorn.run(
        "barkprints.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
