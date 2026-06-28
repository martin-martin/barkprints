#!/usr/bin/env python3
"""Fetch plain-text extracts of Graz-related German Wikipedia articles.

This is the one Graz source that's both clean (CC BY-SA, attribute) and fully
scriptable, so it seeds the corpus immediately. It uses only the stdlib over
the public MediaWiki API; no API key needed. Be polite: low volume, identifies
itself in the User-Agent.

Usage:
    python fetch_wikipedia.py -o raw/wikipedia.txt
    python fetch_wikipedia.py -o raw/wikipedia.txt --extra "Grazer Oper" "Murpark"

Output is appended-friendly plain prose; run clean_text.py on it afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://de.wikipedia.org/w/api.php"
UA = "BarkprintsCorpusBuilder/1.0 (art project; contact: breuss.martin@gmail.com)"

# Graz, its 17 districts, and the landmarks/institutions that give the corpus a
# recognizably-Graz vocabulary. Extend with --extra.
TITLES = [
    "Graz",
    "Geschichte der Stadt Graz",
    "Grazer Schloßberg",
    "Uhrturm (Graz)",
    "Murinsel",
    "Kunsthaus Graz",
    "Universalmuseum Joanneum",
    "Grazer Dom",
    "Mausoleum (Graz)",
    "Landhaus (Graz)",
    "Grazer Burg",
    "Eggenberg (Graz)",
    "Hauptplatz (Graz)",
    "Herrengasse (Graz)",
    "Mur",
    "Karl-Franzens-Universität Graz",
    "Technische Universität Graz",
    "Oper Graz",
    "Forum Stadtpark",
    "Stadtpark (Graz)",
    "Murpark",
    "Grazer Straßenbahn",
    # Districts (Bezirke)
    "Innere Stadt (Graz)",
    "St. Leonhard (Graz)",
    "Geidorf",
    "Lend (Graz)",
    "Gries (Graz)",
    "Jakomini",
    "Liebenau (Graz)",
    "Sankt Peter (Graz)",
    "Waltendorf",
    "Ries (Graz)",
    "Mariatrost",
    "Andritz (Graz)",
    "Gösting",
    "Eggenberg (Grazer Bezirk)",
    "Wetzelsdorf",
    "Straßgang",
    "Puntigam",
]


def fetch_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "extract" in page and page["extract"].strip():
            return page["extract"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True, help="Output raw text file")
    ap.add_argument("--extra", nargs="*", default=[], help="Additional article titles")
    args = ap.parse_args()

    titles = TITLES + args.extra
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    ok = 0
    for t in titles:
        try:
            extract = fetch_extract(t)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  ! {t}: {e}")
            continue
        if extract:
            parts.append(extract)
            ok += 1
            print(f"  + {t} ({len(extract):,} chars)")
        else:
            print(f"  - {t}: no extract")
        time.sleep(0.3)  # be polite

    out.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"\nFetched {ok}/{len(titles)} articles -> {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
