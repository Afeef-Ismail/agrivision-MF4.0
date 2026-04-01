# weather.py
# Fetches current weather data for the farmer's location using OpenWeatherMap free API.
# Uses this data to provide a spray timing recommendation.
#
# Spray timing logic:
# - High humidity (>80%) → spray early morning when humidity drops
# - Rain forecast → warn against spraying (pesticide washoff)
# - Wind speed >15 km/h → warn against spraying (drift risk)
# - Ideal conditions → confirm it's a good time to spray

import os
import requests


# OpenWeatherMap free API base URL
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# deep-translator language codes verified to match Google Translate ISO 639-1 codes.
# 'or' (Odia) is supported by Google Translate — falls back to English on any error.
_TRANSLATE_SUPPORTED = {"hi", "kn", "ml", "mr", "ta", "bn", "gu", "or", "pa", "te"}


def _translate_text(text: str, lang_code: str) -> str:
    """
    Translates text to the target language using deep-translator (free, no API key,
    uses requests — no httpx dependency conflict with groq).
    Returns the original English text on any error so the app never crashes.
    """
    if lang_code == "en" or lang_code not in _TRANSLATE_SUPPORTED:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=lang_code).translate(text)
    except Exception as e:
        print(f"[weather] Translation to '{lang_code}' failed ({e}). Using English.")
        return text


def get_spray_timing(location: str, language_code: str = "en") -> dict:
    """
    Fetches real-time weather for the given location and returns spray timing advice.

    Args:
        location: user-entered location string, e.g. "Mangalore, Karnataka"

    Returns a dict with:
        - "temperature": float (°C)
        - "humidity": int (%)
        - "wind_speed": float (km/h)
        - "description": str — weather condition, e.g. "light rain"
        - "advice": str — human-readable spray timing recommendation
        - "safe_to_spray": bool — True if conditions are suitable
        - "error": str or None — error message if the API call failed
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # If no API key is configured, return a safe fallback message
    if not api_key:
        advice = "Weather API not configured. Check conditions manually before spraying."
        advice = _translate_text(advice, language_code)
        return {
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "description": "Weather data unavailable",
            "advice": advice,
            "safe_to_spray": None,
            "error": "OPENWEATHER_API_KEY not set"
        }

    try:
        # Call OpenWeatherMap current weather endpoint
        # units=metric → temperature in Celsius, wind speed in m/s
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(OWM_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Extract relevant weather fields from the API response
        temperature = data["main"]["temp"]           # °C
        humidity = data["main"]["humidity"]           # %
        wind_speed_ms = data["wind"]["speed"]         # m/s from API
        wind_speed_kmh = round(wind_speed_ms * 3.6, 1)  # convert to km/h

        # Get weather condition description (e.g. "light rain", "clear sky")
        description = data["weather"][0]["description"] if data.get("weather") else "unknown"

        # --- Spray Timing Logic ---
        # Rule 1: Rain → do not spray (pesticide washes off)
        is_raining = any(
            keyword in description.lower()
            for keyword in ["rain", "drizzle", "shower", "storm", "thunderstorm"]
        )

        # Rule 2: High wind → do not spray (spray drift is dangerous and wasteful)
        high_wind = wind_speed_kmh > 15

        # Rule 3: Very high humidity → spray early morning (better absorption when dew dries)
        high_humidity = humidity > 80

        # Determine advice based on conditions
        if is_raining:
            advice = (
                f"Do NOT spray now — {description} detected. "
                "Rain will wash off pesticides and waste your inputs. "
                "Wait until at least 24 hours after rain stops."
            )
            safe_to_spray = False

        elif high_wind:
            advice = (
                f"Do NOT spray now — wind speed is {wind_speed_kmh} km/h, which is too high. "
                "Spray drift can harm nearby crops and is a health hazard. "
                "Spray when wind is below 15 km/h, ideally early morning or evening."
            )
            safe_to_spray = False

        elif high_humidity:
            advice = (
                f"Humidity is high ({humidity}%). Spray early morning (before 8 AM) "
                "when humidity drops slightly. Avoid spraying at night. "
                f"Current temperature: {temperature}°C."
            )
            safe_to_spray = True

        else:
            advice = (
                f"Conditions are suitable for spraying. "
                f"Temperature: {temperature}°C, Humidity: {humidity}%, "
                f"Wind: {wind_speed_kmh} km/h. "
                "Best time is early morning or late afternoon to avoid evaporation."
            )
            safe_to_spray = True

        # Translate advice into the farmer's language if requested
        advice = _translate_text(advice, language_code)

        return {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed_kmh,
            "description": description.capitalize(),
            "advice": advice,
            "safe_to_spray": safe_to_spray,
            "error": None
        }

    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to weather service. Check your internet connection."
    except requests.exceptions.Timeout:
        error_msg = "Weather service timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"Location '{location}' not found. Try a city name like 'Mangalore, IN'."
        elif e.response.status_code == 401:
            error_msg = "Invalid OpenWeatherMap API key. Check your OPENWEATHER_API_KEY."
        else:
            error_msg = f"Weather API error: {str(e)}"
    except Exception as e:
        error_msg = f"Unexpected error fetching weather: {str(e)}"

    # Return a fallback response with the error message
    return {
        "temperature": None,
        "humidity": None,
        "wind_speed": None,
        "description": "Unavailable",
        "advice": f"Weather data unavailable: {error_msg}",
        "safe_to_spray": None,
        "error": error_msg
    }
