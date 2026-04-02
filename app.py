# app.py
# Main FastAPI application for AgriVision.
# Handles all HTTP routes and orchestrates the full prediction pipeline:
#   Upload → Validate → Predict → Severity → LLM Recommendation → Voice → Weather → Response

import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path

# Load environment variables from .env BEFORE any module that calls os.getenv()
# (e.g. llm.py reads GROQ_API_KEY, weather.py reads OPENWEATHER_API_KEY)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# Import our pipeline modules.
# IMPORTANT: predict_disease is aliased to run_prediction to avoid a name
# collision with the FastAPI route handler defined below (also named predict_*).
# Without the alias, defining the route handler would shadow the imported
# function, causing a recursive call → TypeError at runtime.
from predict import load_model, predict_disease as run_prediction, is_model_loaded
from validate import check_blur, check_confidence
from severity import get_severity
from disease_info import get_neighbouring_risk
from llm import get_recommendation
from voice import generate_voice, LANGUAGE_CODES
from weather import get_spray_timing
from gps_validator import get_gps_warning
from report_generator import generate_txt_report, generate_pdf_report


# ---------------------------------------------------------------------------
# Translation helpers — used to localise the entire JSON response when
# lang_code != 'en'. Falls back silently to the original English text on
# any error so the app never crashes due to a translation failure.
# ---------------------------------------------------------------------------
def _translate(text: str, lang_code: str) -> str:
    if not text or not isinstance(text, str) or lang_code == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=lang_code).translate(text)
    except Exception:
        return text


def _translate_list(items: list, lang_code: str) -> list:
    return [_translate(item, lang_code) for item in (items or [])]


def _translate_recommendation(rec: dict, lang_code: str) -> dict:
    """Translates all string values in the LLM recommendation dict."""
    if lang_code == "en":
        return rec
    return {
        "immediate_actions":    _translate_list(rec.get("immediate_actions", []), lang_code),
        "treatment":            _translate_list(rec.get("treatment", []), lang_code),
        "recovery_time":        _translate(rec.get("recovery_time", ""), lang_code),
        "preventive_measures":  _translate_list(rec.get("preventive_measures", []), lang_code),
        "neighbouring_crop_risk": _translate(rec.get("neighbouring_crop_risk", ""), lang_code),
    }

# --- App Initialization ---
app = FastAPI(
    title="AgriVision",
    description="AI Crop Disease Detection for Indian Farmers",
    version="1.0.0"
)

# Allow all origins for CORS — required for cross-origin requests during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, uploaded images, generated audio files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template engine for serving index.html
templates = Jinja2Templates(directory="templates")

# Ensure upload and audio directories exist before any request comes in
UPLOAD_DIR = Path("static/uploads")
AUDIO_DIR = Path("static/audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    """
    Called once when the FastAPI server starts.
    Tries to load the ML model. If the model file is missing, the app
    still starts — /predict will return a friendly JSON error instead of crashing.
    """
    print("[app] AgriVision starting up...")
    success = load_model()
    if success:
        print("[app] ML model loaded. Ready for inference.")
    else:
        print("[app] WARNING: ML model not loaded.")
        print("[app] Add the model file to: model/agrivision_efficientnetb3_final.h5")
        print("[app] The server is running but /predict will return an error until the model is present.")


# --- Routes ---

@app.get("/")
async def serve_index(request: Request):
    """Serves the main single-page frontend (index.html)."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Could not load frontend: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint — returns server status and model load state.
    Judges/testers can call this to confirm the server is running.
    """
    try:
        return {
            "status": "ok",
            "model_loaded": is_model_loaded(),
            "message": "AgriVision is running"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    crop_type: str = Form(...),
    location: str = Form(default="Mangalore, Karnataka"),
    language: str = Form(default="English"),
    mode: str = Form(default="chemical")
):
    """
    Main prediction endpoint. Accepts a leaf image and farm details.

    Full pipeline:
    1. Save uploaded image to disk
    2. Validate: blur check (Laplacian variance, threshold 100)
    3. Run ML model inference (EfficientNet-B3)
    4. Validate: confidence check (minimum 60%)
    5. Get severity and traffic light color
    6. Get neighbouring crop risk warning
    7. Get LLM agronomic recommendation (Groq / fallback dict)
    8. Generate voice audio file (gTTS)
    9. Get weather-based spray timing advice (OpenWeatherMap)
    10. Return everything as a single JSON response

    Args:
        image: uploaded leaf image (JPEG/PNG)
        crop_type: "Tomato", "Apple", or "Grape"
        location: farm location string for weather lookup
        language: voice output language (English/Hindi/Kannada/Malayalam/Marathi/Tamil)
        mode: treatment preference — "organic" or "chemical"
    """
    # Wrap the entire route in try/except so no unhandled exception ever
    # reaches the client as a 500 HTML error page
    try:
        # Generate a unique filename to avoid collisions between simultaneous uploads
        unique_id = str(uuid.uuid4())[:8]
        original_ext = Path(image.filename).suffix if image.filename else ".jpg"
        image_filename = f"{unique_id}{original_ext}"
        image_path = UPLOAD_DIR / image_filename

        # Save the uploaded image to disk
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # --- GPS Check (runs immediately after save, before any heavy processing) ---
        # Uses only Pillow EXIF — no external API. Returns None if no GPS in image.
        gps_info = get_gps_warning(str(image_path))

        # --- Step 1: Check if model is loaded before doing any work ---
        # Return a helpful message immediately rather than processing the image
        # and failing later — saves time and gives a clear error
        if not is_model_loaded():
            os.remove(image_path)
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Model not loaded yet. Please add model/agrivision_efficientnetb3_final.h5"
                }
            )

        # --- Step 2: Blur Validation ---
        is_sharp, blur_message = check_blur(str(image_path))
        if not is_sharp:
            os.remove(image_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": blur_message}
            )

        # --- Step 3: ML Model Inference ---
        # run_prediction is predict.predict_disease imported with an alias above.
        # The alias prevents this route handler from shadowing the imported function.
        prediction_result = run_prediction(str(image_path), crop_type)

        if not prediction_result["success"]:
            return JSONResponse(
                status_code=503,
                content={"success": False, "error": prediction_result["error"]}
            )

        class_name = prediction_result["class_name"]
        disease_readable = prediction_result["disease"]
        confidence = prediction_result["confidence"]
        prediction_warning = prediction_result.get("warning")

        # --- Step 4: Confidence Validation ---
        is_confident, confidence_message = check_confidence(confidence)
        if not is_confident:
            os.remove(image_path)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": confidence_message,
                    "confidence": confidence
                }
            )

        # --- Step 5: Severity Mapping ---
        severity_data = get_severity(class_name)
        severity = severity_data["severity"]
        color = severity_data["color"]
        disease_display = severity_data["display"]

        # --- Step 6: Neighbouring Crop Risk ---
        # Passed as context into the LLM prompt — not returned directly in the response.
        neighbouring_risk = get_neighbouring_risk(class_name)

        # --- Step 7: LLM Recommendation (always English, translated below) ---
        hour = datetime.now().hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        crop_from_class = class_name.split("___")[0] if "___" in class_name else crop_type

        recommendation = get_recommendation(
            crop=crop_from_class,
            disease=disease_display,
            severity=str(severity) if severity else "None (Healthy)",
            location=location,
            time_of_day=time_of_day,
            mode=mode,
            neighbouring_risk=neighbouring_risk   # feeds disease spread context into the prompt
        )

        # --- Step 8: Weather & Spray Timing ---
        # Done BEFORE voice so the translated advice can be spoken in the audio.
        lang_code = LANGUAGE_CODES.get(language, "en")
        weather_data = get_spray_timing(location, lang_code)

        # --- Translate all response text to the farmer's language ---
        # LLM always returns English; deep-translator localises the response.
        recommendation = _translate_recommendation(recommendation, lang_code)
        disease_display_translated = _translate(disease_display, lang_code)
        gps_info_translated = _translate(gps_info, lang_code) if gps_info else None

        # --- Step 9: Voice Output ---
        # Combines disease recommendation + translated weather advice into one MP3.
        audio_filename = f"advice_{unique_id}.mp3"
        audio_path = str(AUDIO_DIR / audio_filename)

        voice_success, voice_result = generate_voice(
            recommendation=recommendation,
            disease_display=disease_display,
            severity=str(severity) if severity else "None",
            language=language,
            output_path=audio_path,
            weather_advice=weather_data.get("advice")
        )

        # Return relative URL for the frontend audio player (/static/ is mounted above)
        audio_url = f"/static/audio/{audio_filename}" if voice_success else None

        # --- Step 10: Return Final Response ---
        # Shape matches displayResults() in index.html exactly:
        #   data.prediction.disease_display / data.prediction.confidence / data.prediction.warning
        #   data.severity.color / data.severity.level
        #   data.weather  (full object — frontend reads temperature, humidity, etc.)
        #   data.audio_url / data.gps_info
        return JSONResponse(content={
            "success": True,
            "crop_type": crop_type,          # included so the report knows the crop
            "prediction": {
                "disease_display": disease_display_translated,
                "confidence": confidence,
                "warning": prediction_warning    # None or moderate-confidence warning string
            },
            "severity": {
                "color": color,
                "level": severity          # None for healthy, "Low"/"Moderate"/"High"
            },
            "recommendation": recommendation,  # all text fields translated
            "weather": weather_data,           # advice already translated by weather.py
            "audio_url": audio_url,
            "gps_info": gps_info_translated    # None | "✅ GPS: ..." | "⚠️ GPS Warning: ..."
            # neighbouring_risk removed from direct response — lives inside recommendation dict
        })

    except Exception as e:
        # Catch-all: return a clean JSON error instead of a 500 HTML stack trace
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}"
            }
        )


@app.post("/download/txt")
async def download_txt(data: dict):
    """
    Generates and returns a plain-text crop disease report as a file download.

    Accepts the full prediction response dict (as JSON body) that the frontend
    stores after a successful /predict call. Streams the report back with
    Content-Disposition: attachment so the browser triggers a Save dialog.

    Args:
        data: prediction response dict (crop_type, prediction, severity,
              recommendation, weather, gps_info)
    """
    try:
        txt = generate_txt_report(data)
        return Response(
            content=txt.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=agrivision_report.txt"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/download/pdf")
async def download_pdf(data: dict):
    """
    Generates and returns a PDF crop disease report as a file download.

    Writes the PDF to a temporary path inside static/reports/, returns it as
    a FileResponse (which FastAPI streams with proper headers), then schedules
    the file for deletion via a background task to keep the directory clean.

    Args:
        data: prediction response dict (same shape as /download/txt)
    """
    import uuid
    from starlette.background import BackgroundTask

    try:
        reports_dir = Path("static/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = str(reports_dir / f"report_{uuid.uuid4().hex[:8]}.pdf")
        generate_pdf_report(data, pdf_path)

        # Delete the temp PDF file after it has been streamed to the client
        def _cleanup(path: str):
            try:
                os.remove(path)
            except Exception:
                pass

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="agrivision_report.pdf",
            background=BackgroundTask(_cleanup, pdf_path)
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
