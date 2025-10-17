# Multimeta Dataset Analyzer (Flask)

A Flask backend that allows users to upload a CSV/Excel file, computes basic statistics with pandas/numpy, generates charts with matplotlib, and produces AI insights via OpenAI (or a rule-based fallback).

## Features
- File upload for CSV/XLS/XLSX stored under `uploads/`
- Stats: mean, median, mode, std for numeric columns
- Charts: bar (means), line (first numeric col), pie (top categories)
- AI summary using OpenAI; fallback summary if no API key
- Results rendered via HTML templates

## Requirements
- Python 3.10+

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create the folders if not present:
```bash
mkdir uploads static
```

(Optional) set your OpenAI API key:
```bash
setx OPENAI_API_KEY "your_api_key_here"   # Windows
# macOS/Linux: export OPENAI_API_KEY=your_api_key_here
```

Run locally:
```bash
python app.py
```
Open `http://localhost:5000/` and upload a dataset.

## Environment Variables
- `OPENAI_API_KEY` (optional): enables AI-generated insights
- `OPENAI_MODEL` (optional): defaults to `gpt-4o-mini`
- `FLASK_SECRET_KEY` (optional): Flask secret key for sessions
- `PORT` (optional): port to bind (used by platforms like Railway/Heroku)

## Deploy (Heroku/Railway)
- Ensure `Procfile`, `requirements.txt` are present
- Set `OPENAI_API_KEY` in project settings to enable AI insights
- The app binds to `$PORT` automatically

## Notes
- Charts are embedded as base64 images on the result page
- Upload size limit: 20 MB (configurable via `MAX_CONTENT_LENGTH_MB` in `app.py`)
- If OpenAI is unavailable, a rule-based textual summary is shown

