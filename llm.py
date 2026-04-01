# llm.py
# Generates structured agronomic recommendations using the Groq API.
# Model: llama-3.1-8b-instant (replaces decommissioned llama3-8b-8192)
# The prompt instructs the LLM to respond in the farmer's chosen language so
# the voice output (gTTS) speaks naturally in that language.

import os
import json
import re

# Groq client imported inside get_recommendation() to avoid a top-level crash
# if the package is not installed.


def _build_prompt(crop: str, disease: str, severity: str, location: str,
                  time_of_day: str, mode: str,
                  neighbouring_risk: str = "") -> str:
    """
    Builds a structured prompt that instructs the LLM to respond ONLY in JSON (English).
    Translation to the farmer's language is handled by app.py after this call.

    - JSON schema is fixed so _parse_llm_response() can reliably parse it
    - mode controls organic vs chemical treatment suggestions
    - neighbouring_risk provides static disease-spread context as extra LLM input
    """
    mode_instruction = (
        "Use only ORGANIC treatments (neem oil, copper fungicide, compost teas, biological controls)."
        if mode == "organic"
        else "You may suggest CHEMICAL treatments (fungicides, pesticides, herbicides) as needed."
    )

    risk_context = (
        f"\nNeighbouring crop risk context (use this to inform your neighbouring_crop_risk field): "
        f"{neighbouring_risk}"
        if neighbouring_risk
        else ""
    )

    prompt = f"""You are an expert agronomist advising an Indian farmer.

Crop: {crop}
Disease Detected: {disease}
Severity: {severity if severity else "None (Healthy plant)"}
Farm Location: {location}
Time of Day: {time_of_day}
Treatment Mode: {mode_instruction}{risk_context}

Provide a complete disease management plan. Respond ONLY with valid JSON — no preamble, no explanation, no markdown, no extra text.

Return exactly this JSON structure:
{{
  "immediate_actions": ["step 1", "step 2", "step 3"],
  "treatment": ["treatment 1", "treatment 2", "treatment 3"],
  "recovery_time": "estimated recovery period as a string",
  "preventive_measures": ["measure 1", "measure 2", "measure 3"],
  "neighbouring_crop_risk": "one sentence about risk to nearby crops"
}}

Keep each item concise and practical. Tailor advice to Indian farming conditions. Respond in English."""

    return prompt


def _parse_llm_response(response_text: str) -> dict:
    """
    Parses the LLM JSON response, stripping any markdown code fences first.
    Returns a safe fallback dict if parsing fails.
    """
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip().strip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "immediate_actions": ["Isolate affected plants immediately."],
            "treatment": ["Consult a local agricultural officer for guidance."],
            "recovery_time": "Unknown — severity assessment required.",
            "preventive_measures": ["Monitor plants daily.", "Maintain field hygiene."],
            "neighbouring_crop_risk": "Unable to assess risk at this time."
        }


def get_recommendation(crop: str, disease: str, severity: str, location: str,
                       time_of_day: str, mode: str,
                       neighbouring_risk: str = "") -> dict:
    """
    Generates a structured agronomic recommendation for the detected disease.
    Always returns English — app.py handles translation to the farmer's language.

    Uses Groq (llama-3.1-8b-instant) for all generated recommendations.
    If Groq fails (network error, rate limit, missing key), returns a safe
    English fallback dict.

    Args:
        crop: e.g. "Tomato"
        disease: e.g. "Tomato Late Blight"
        severity: e.g. "High"
        location: e.g. "Mangalore, Karnataka"
        time_of_day: "morning", "afternoon", or "evening"
        mode: "organic" or "chemical"
        neighbouring_risk: disease spread context from disease_info.py

    Returns:
        Dict with keys: immediate_actions, treatment, recovery_time,
        preventive_measures, neighbouring_crop_risk
    """
    prompt = _build_prompt(crop, disease, severity, location, time_of_day, mode,
                           neighbouring_risk)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # replaces decommissioned llama3-8b-8192
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            response_text = response.choices[0].message.content
            return _parse_llm_response(response_text)

        except Exception as groq_error:
            print(f"[llm] Groq failed: {groq_error}. Returning fallback recommendation.")

    print("[llm] Groq unavailable or key missing. Returning fallback recommendation.")
    return {
        "immediate_actions": [
            "Remove and destroy visibly infected plant parts.",
            "Isolate affected plants from healthy ones.",
            "Avoid overhead irrigation to reduce moisture spread."
        ],
        "treatment": [
            "Apply a broad-spectrum fungicide or bactericide as appropriate.",
            "Consult your local Krishi Vigyan Kendra (KVK) for approved products.",
        ],
        "recovery_time": "2–4 weeks with proper treatment.",
        "preventive_measures": [
            "Practice crop rotation every season.",
            "Use certified disease-resistant seed varieties.",
            "Maintain proper plant spacing for airflow."
        ],
        "neighbouring_crop_risk": (
            "Monitor neighbouring crops for similar symptoms and take preventive action."
        )
    }
