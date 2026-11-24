#!/usr/bin/env bash
set -e

# Change directory to project root
cd "$(dirname "$0")"

echo "==============================================================================="
echo "  Harmful Brain Activity Classification & EEG Seizure Detection Prototype"
echo "  macOS / Linux Launcher"
echo "==============================================================================="

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Make sure newly added dependencies are present
python -c "import fastapi, uvicorn, multipart" 2>/dev/null || pip install -r requirements.txt

echo "Starting Harmful Brain Activity Classification on http://localhost:8501 ..."
python run_app.py --port 8501
