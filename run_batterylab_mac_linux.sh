#!/usr/bin/env bash
set -e
echo "🔋 BatteryLab Launcher (macOS/Linux)"
echo "Creating Python environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "Installing requirements (this may take a minute)..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt
echo "Launching BatteryLab in your browser..."
streamlit run app.py
