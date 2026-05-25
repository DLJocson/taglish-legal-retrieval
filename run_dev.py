#!/usr/bin/env python3
"""
Development server with live frontend reload (no server restart needed for UI edits).

- Sets DEV=1 so HTML/CSS/JS are served with no-cache headers (refresh browser to see changes).
- Does not use --reload by default (avoids reloading heavy embedding models on every save).

Usage:
  python run_dev.py
  python run_dev.py --reload-backend   # also restart on Python file changes
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DEV", "1")


def main() -> None:
    import uvicorn

    reload = "--reload-backend" in sys.argv
    kwargs: dict = {
        "app": "api:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": reload,
    }
    if reload:
        kwargs["reload_dirs"] = ["backend", "frontend"]

    print("PH Legal AI Search — dev server")
    print("  Search:     http://127.0.0.1:8000/")
    print("  Analytics:  http://127.0.0.1:8000/analytics")
    print("  DEV=1: save CSS/JS/HTML, then refresh the browser (no server restart).")
    if reload:
        print("  --reload-backend: Python changes restart the server (models reload).")
    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
