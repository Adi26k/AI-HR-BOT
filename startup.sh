#!/bin/bash
cd /home/site/wwwroot/

echo "Python version:"
python --version

echo "Installing core packages first..."
pip install --user uvicorn fastapi

echo "Setting Python path..."
export PYTHONPATH=/home/site/.local/lib/python3.13/site-packages:$PYTHONPATH

echo "Installing other requirements..."
# Try to install other requirements but continue even if some fail
pip install --user -r requirements.txt || echo "Some packages could not be installed but continuing..."

echo "Verifying uvicorn installation:"
pip list | grep uvicorn

echo "Starting application..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000