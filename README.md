# AgriVision

AI Crop Disease Detection Web App (Tomato / Apple / Grape)

## Overview

AgriVision is a local FastAPI app for field agronomists and farmers. It detects crop disease from leaf images and returns:
- disease name + confidence
- severity traffic light
- LLM-based treatment plan (Groq)
- voice alert (gTTS) in local language
- spray timing advice (OpenWeatherMap)
- neighbouring crop risk warning

## File summary

- `app.py`: FastAPI routes, full prediction pipeline
- `predict.py`: EfficientNet-B3 model loading + inference
- `validate.py`: blur check + confidence gate
- `llm.py`: Groq structured JSON recommendations
- `voice.py`: gTTS MP3 generation for multi-lingual output
- `weather.py`: OpenWeatherMap spray timing advisor
- `severity.py`: 18-class severity + traffic light map
- `disease_info.py`: neighbour risk warnings by disease
- `gps_validator.py`: GPS EXIF validation for Indian farming regions
- `templates/index.html`: UI for farmers (single page)
- `static/style.css`: responsive farm-friendly styling
- `static/uploads/`: temp image uploads (gitignored)
- `static/audio/`: voice files (gitignored)

## Prerequisites

1. Python 3.11+ (3.12 works)
2. Create and activate a virtual environment (recommended)

## Install dependencies

```bash
# Clone the repo and cd into it
git clone https://github.com/Afeef-Ismail/agrivision-MF4.0.git
cd agrivision-MF4.0

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Install ffmpeg (required for pydub audio with inter-segment pauses)

```bash
# Linux / WSL
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# and add the bin/ folder to your PATH environment variable.
# Without ffmpeg the app still works — audio is generated without pauses.
```

## Add model weights

1. Place `agrivision_efficientnetb3_final.h5` in `model/`
2. `model/class_indices.json` is already included in the repo

> **Note:** The `.h5` model weight file (~74 MB) is excluded from the repo via `.gitignore`.
> Download it separately or train your own using the EfficientNet-B3 architecture described in `predict.py`.

## Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENWEATHER_API_KEY` — [OpenWeatherMap free tier](https://openweathermap.org/api)
- `GROQ_API_KEY` — [Groq free tier](https://console.groq.com/)

## Run the app

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open in browser: `http://127.0.0.1:8000`

## Health check

`GET /health` returns `status` and `model_loaded`.

## Notes

- If model is missing, API returns clear message: "Model not loaded yet".
- No hardcoded predictions are used; inference is real when model exists.
- LLM recommendation is dynamic and structured JSON.
- For judges: upload unknown test images, ensure model file is loaded.
