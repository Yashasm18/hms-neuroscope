"""
Start the Harmful Brain Activity Classification web app.

    python run_app.py                 # http://localhost:8501, opens your browser
    python run_app.py --port 8502     # another port
    python run_app.py --no-browser    # do not open a browser tab

Hosting platforms (Render etc.) set $PORT; the app then listens on 0.0.0.0.
"""

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Harmful Brain Activity Classification web app")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8501)))
    parser.add_argument("--host", default="0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    import uvicorn

    url = f"http://localhost:{args.port}"
    if not args.no_browser and "PORT" not in os.environ:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    print(f"\n  Harmful Brain Activity Classification is starting at {url}\n  Press Ctrl+C to stop.\n")
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
