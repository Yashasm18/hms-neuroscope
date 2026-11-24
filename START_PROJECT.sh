#!/usr/bin/env bash
set -e

# Change directory to project root
cd "$(dirname "$0")"

echo "==============================================================================="
echo "  HMS - Harmful Brain Activity Classifier & EEG Seizure Detection Prototype"
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

echo "Starting Streamlit on http://localhost:8501 ..."
streamlit run app.py --server.port=8501
