# 🌿 AgriVision

**AI Crop Disease Detection Web App for Indian Farmers**
Crops supported: Tomato · Apple · Grape

---

## Overview

AgriVision is a FastAPI-based web application that detects crop diseases from leaf images using an EfficientNet-B3 deep learning model trained on the PlantVillage dataset. It provides:

- 🔬 **Disease detection** with confidence score (18 classes)
- 🚦 **Severity traffic light** — Green / Yellow / Red
- 💊 **LLM treatment plan** — structured JSON via Groq (organic or chemical mode)
- 🔊 **Voice advisory** — gTTS audio in 11 Indian languages
- 🌤 **Spray timing advisor** — real-time weather from OpenWeatherMap
- ⚠️ **Neighbouring crop risk** — alerts about disease spread to nearby fields
- 📍 **GPS validation** — EXIF-based farm region check (no external API)

---

## Project Structure

```
agrivision-MF4.0/
├── app.py                  # FastAPI routes + full prediction pipeline
├── predict.py              # EfficientNet-B3 model loading + TTA inference
├── validate.py             # Blur detection + confidence gate
├── llm.py                  # Groq LLM structured recommendations
├── voice.py                # gTTS multi-lingual voice generation (11 languages)
├── weather.py              # OpenWeatherMap spray timing advisor
├── severity.py             # 18-class severity + traffic light mapping
├── disease_info.py         # Neighbouring crop risk warnings
├── gps_validator.py        # GPS EXIF validation for Indian farm regions
│
├── model/
│   ├── class_indices.json              # 18 class labels (included in repo)
│   └── agrivision_efficientnetb3_final.h5  # Model weights (~74 MB, see below)
│
├── templates/
│   └── index.html          # Single-page frontend UI
│
├── static/
│   ├── style.css           # Responsive farm-friendly styling
│   ├── uploads/            # Temp image uploads (auto-created, gitignored)
│   └── audio/              # Generated voice files (auto-created, gitignored)
│
├── requirements.txt        # Python dependencies
├── .env.example            # Template for API keys
├── .gitignore
└── README.md
```

---

## Quick Start (for Judges)

### 1. Clone and set up environment

```bash
git clone https://github.com/Afeef-Ismail/agrivision-MF4.0.git
cd agrivision-MF4.0

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add model weights

Place the trained model file in the `model/` directory:

```
model/agrivision_efficientnetb3_final.h5
```

> **Note:** The `.h5` file (~74 MB) is excluded from the repo via `.gitignore`.
> It will be provided separately during the demo / submission.

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your free-tier keys:

| Variable | Source | Free? |
|----------|--------|-------|
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) | ✅ Yes |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) | ✅ Yes |

### 4. (Optional) Install ffmpeg

Required only for inter-segment pauses in voice output. The app works without it.

```bash
# Linux / WSL
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# and add the bin/ folder to your PATH.
```

### 5. Run the app

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open in browser: **http://127.0.0.1:8000**

### 6. Health check

```
GET /health
```

Returns `{ "status": "ok", "model_loaded": true }` when everything is ready.

---

## How It Works

```
Upload Image → Blur Check → EfficientNet-B3 (TTA) → Confidence Gate
                                    ↓
                            Disease + Severity
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              LLM Advice      Weather API      GPS Check
              (Groq)        (OpenWeatherMap)    (EXIF)
                    ↓               ↓
                Voice (gTTS) ← Spray Timing
                    ↓
              JSON Response → Frontend
```

---

## Supported Languages (Voice + UI)

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Bengali | `bn` |
| Hindi | `hi` | Gujarati | `gu` |
| Kannada | `kn` | Odia | `or` |
| Malayalam | `ml` | Punjabi | `pa` |
| Marathi | `mr` | Telugu | `te` |
| Tamil | `ta` | | |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| ML Model | EfficientNet-B3 (TensorFlow/Keras) |
| LLM | Groq (Llama 3.1 8B Instant) |
| Voice | gTTS + pydub |
| Weather | OpenWeatherMap API |
| Translation | deep-translator (Google Translate) |
| Frontend | Vanilla HTML/CSS/JS (Jinja2 template) |

---

## Notes for Judges

- **No hardcoded predictions** — all inference is real via the loaded model.
- **LLM recommendations are dynamic** — structured JSON generated per request.
- **Test with unknown images** — upload any tomato/apple/grape leaf photo.
- **Health endpoint** — call `/health` to confirm `model_loaded: true` before testing.
- **Graceful fallbacks** — app runs even without API keys or model (returns clear error messages).
