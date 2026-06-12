#!/usr/bin/env python3
"""
Development server with hot-reloading.

Frontend changes (HTML/CSS/JS): Refresh browser - no restart needed.
Backend changes (Python): Auto-reload with uvicorn (models reload on change).

Usage:
  python run_dev.py                    # Auto-reload on all changes
  python run_dev.py --no-reload        # No auto-reload (manual restart)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DEV", "1")


def main() -> None:
    import uvicorn

    no_reload = "--no-reload" in sys.argv
    reload = not no_reload

    kwargs: dict = {
        "app": "api:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": reload,
        "reload_dirs": ["backend", "frontend"] if reload else None,
    }

    print("PH Legal AI Search — dev server")
    print("  Search:     http://127.0.0.1:8000/")
    print("  Analytics:  http://127.0.0.1:8000/analytics")
    print("\n  Hot-reload status:")
    if reload:
        print("  ✓ Frontend: Save HTML/CSS/JS, then refresh browser")
        print("  ✓ Backend:  Auto-restarts on Python changes (models reload)")
    else:
        print("  ✗ Auto-reload disabled (manual restart required)")
    print("\n  Press Ctrl+C to stop\n")

    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
