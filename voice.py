# voice.py
# Converts the LLM recommendation to speech using gTTS.
# gTTS does NOT translate — it reads whatever text you give it in the
# specified language's voice. So the text must already be in the target
# language. This is achieved by:
#   1. llm.py generates recommendation content in the farmer's language
#   2. _build_voice_segments() uses translated framing phrases (defined here)
#   3. weather.py translates the spray advice via googletrans
#   4. gTTS reads the assembled text in the correct language
#
# Audio continuity: if pydub + ffmpeg are installed, each spoken point is
# a separate mp3 segment with a 0.5 s silence inserted between them.
# This gives the farmer time to process each step before the next begins.
# Falls back to a single-pass gTTS call if pydub is unavailable.

import io
import os
import re
from gtts import gTTS

# gTTS language codes confirmed to work; others fall back to English.
# Punjabi ('pa') falls back to Hindi ('hi') if unsupported.
# Odia ('or') falls back to English if unsupported.
GTTS_SUPPORTED = ['en', 'hi', 'pa', 'gu', 'bn', 'or', 'te', 'kn', 'ml', 'mr', 'ta']

# pydub is used for inter-segment silence. Requires ffmpeg to be installed.
# Gracefully degraded: if not available, the app still generates audio —
# just without pauses between each spoken point.
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False
    print("[voice] pydub not installed. Audio will be generated without inter-segment pauses.")

# Maps UI language name → gTTS language code
LANGUAGE_CODES = {
    "Bengali":   "bn",
    "English":   "en",
    "Gujarati":  "gu",
    "Hindi":     "hi",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Marathi":   "mr",
    "Odia":      "or",
    "Punjabi":   "pa",
    "Tamil":     "ta",
    "Telugu":    "te",
}

# Framing phrases used when assembling the spoken text.
# Each entry is translated so the entire audio stays in one language.
# Keys match the gTTS language codes.
VOICE_FRAMING = {
    "en": {
        "disease":       "Disease detected: {disease}. Severity: {severity}.",
        "healthy":       "Your crop appears healthy. No disease detected.",
        "actions":       "Immediate actions to take.",
        "step":          "Step {n}.",
        "treatment":     "Recommended treatment.",
        "recovery":      "Expected recovery time: {recovery}",
        "prevention":    "To prevent future outbreaks.",
        "weather_intro": "Weather advisory.",
    },
    "hi": {
        "disease":       "बीमारी पाई गई: {disease}। गंभीरता: {severity}।",
        "healthy":       "आपकी फसल स्वस्थ है। कोई बीमारी नहीं मिली।",
        "actions":       "तत्काल उठाए जाने वाले कदम।",
        "step":          "चरण {n}।",
        "treatment":     "सुझाया गया उपचार।",
        "recovery":      "ठीक होने का अनुमानित समय: {recovery}",
        "prevention":    "भविष्य में बीमारी से बचाव के लिए।",
        "weather_intro": "मौसम की सलाह।",
    },
    "kn": {
        "disease":       "ರೋಗ ಪತ್ತೆಯಾಗಿದೆ: {disease}. ತೀವ್ರತೆ: {severity}.",
        "healthy":       "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ರೋಗ ಇಲ್ಲ.",
        "actions":       "ತಕ್ಷಣ ತೆಗೆದುಕೊಳ್ಳಬೇಕಾದ ಕ್ರಮಗಳು.",
        "step":          "ಹಂತ {n}.",
        "treatment":     "ಶಿಫಾರಸು ಮಾಡಲಾದ ಚಿಕಿತ್ಸೆ.",
        "recovery":      "ಅಂದಾಜು ಚೇತರಿಕೆ ಸಮಯ: {recovery}",
        "prevention":    "ಭವಿಷ್ಯದ ರೋಗ ತಡೆಗಟ್ಟಲು.",
        "weather_intro": "ಹವಾಮಾನ ಸಲಹೆ.",
    },
    "ml": {
        "disease":       "രോഗം കണ്ടെത്തി: {disease}. കാഠിന്യം: {severity}.",
        "healthy":       "നിങ്ങളുടെ വിള ആരോഗ്യകരമാണ്. രോഗം ഇല്ല.",
        "actions":       "ഉടനടി സ്വീകരിക്കേണ്ട നടപടികൾ.",
        "step":          "ഘട്ടം {n}.",
        "treatment":     "ശുപാർശ ചെയ്ത ചികിത്സ.",
        "recovery":      "പ്രതീക്ഷിക്കുന്ന വീണ്ടെടുക്കൽ സമയം: {recovery}",
        "prevention":    "ഭാവിയിൽ രോഗം തടയാൻ.",
        "weather_intro": "കാലാവസ്ഥ ഉപദേശം.",
    },
    "mr": {
        "disease":       "रोग आढळला: {disease}. तीव्रता: {severity}.",
        "healthy":       "तुमचे पीक निरोगी आहे. कोणताही रोग नाही.",
        "actions":       "तात्काळ घ्यायचे उपाय.",
        "step":          "पाऊल {n}.",
        "treatment":     "शिफारस केलेले उपचार.",
        "recovery":      "अपेक्षित बरे होण्याचा वेळ: {recovery}",
        "prevention":    "भविष्यात रोग टाळण्यासाठी.",
        "weather_intro": "हवामान सल्ला.",
    },
    "ta": {
        "disease":       "நோய் கண்டறியப்பட்டது: {disease}. தீவிரம்: {severity}.",
        "healthy":       "உங்கள் பயிர் ஆரோக்கியமாக உள்ளது. நோய் இல்லை.",
        "actions":       "உடனடியாக எடுக்க வேண்டிய நடவடிக்கைகள்.",
        "step":          "படி {n}.",
        "treatment":     "பரிந்துரைக்கப்பட்ட சிகிச்சை.",
        "recovery":      "எதிர்பார்க்கப்படும் குணமடைவு நேரம்: {recovery}",
        "prevention":    "எதிர்கால நோய் தடுக்க.",
        "weather_intro": "வானிலை ஆலோசனை.",
    },
    # --- New Indian languages ---
    "bn": {
        "disease":       "রোগ পাওয়া গেছে: {disease}. তীব্রতা: {severity}.",
        "healthy":       "আপনার ফসল সুস্থ। কোনো রোগ নেই।",
        "actions":       "তাৎক্ষণিক পদক্ষেপ নিন।",
        "step":          "ধাপ {n}।",
        "treatment":     "প্রস্তাবিত চিকিৎসা।",
        "recovery":      "প্রত্যাশিত সুস্থ হওয়ার সময়: {recovery}",
        "prevention":    "ভবিষ্যতে রোগ প্রতিরোধে।",
        "weather_intro": "আবহাওয়া পরামর্শ।",
    },
    "gu": {
        "disease":       "રોગ મળ્યો: {disease}. તીવ્રતા: {severity}.",
        "healthy":       "તમારો પાક સ્વસ્થ છે. કોઈ રોગ નથી.",
        "actions":       "તાત્કાલિક પગલાં લો.",
        "step":          "પગલું {n}.",
        "treatment":     "ભલામણ કરેલ સારવાર.",
        "recovery":      "અપેક્ષિત સ્વસ્થ થવાનો સમય: {recovery}",
        "prevention":    "ભવિષ્યમાં રોગ અટકાવવા.",
        "weather_intro": "હવામાન સલાહ.",
    },
    "or": {
        "disease":       "ରୋଗ ଚିହ୍ନଟ ହୋଇଛି: {disease}. ଗୁରୁତ୍ୱ: {severity}.",
        "healthy":       "ଆପଣଙ୍କ ଫସଲ ସୁସ୍ଥ ଅଛି। କୌଣସି ରୋଗ ନାହିଁ।",
        "actions":       "ତୁରନ୍ତ ପଦକ୍ଷେପ ନିଅନ୍ତୁ।",
        "step":          "ପଦକ୍ଷେପ {n}।",
        "treatment":     "ପ୍ରସ୍ତାବିତ ଚିକିତ୍ସା।",
        "recovery":      "ଆଶା ଥିବା ସୁସ୍ଥ ହେବା ସମୟ: {recovery}",
        "prevention":    "ଭବିଷ୍ୟତ ରୋଗ ପ୍ରତିରୋଧ।",
        "weather_intro": "ପାଣିପାଗ ପରାମର୍ଶ।",
    },
    "pa": {
        "disease":       "ਬਿਮਾਰੀ ਮਿਲੀ: {disease}. ਗੰਭੀਰਤਾ: {severity}.",
        "healthy":       "ਤੁਹਾਡੀ ਫ਼ਸਲ ਤੰਦਰੁਸਤ ਹੈ। ਕੋਈ ਬਿਮਾਰੀ ਨਹੀਂ।",
        "actions":       "ਤੁਰੰਤ ਕਦਮ ਚੁੱਕੋ।",
        "step":          "ਕਦਮ {n}।",
        "treatment":     "ਸਿਫ਼ਾਰਸ਼ ਕੀਤਾ ਇਲਾਜ।",
        "recovery":      "ਠੀਕ ਹੋਣ ਦਾ ਸੰਭਾਵਿਤ ਸਮਾਂ: {recovery}",
        "prevention":    "ਭਵਿੱਖੀ ਬਿਮਾਰੀ ਤੋਂ ਬਚਾਅ।",
        "weather_intro": "ਮੌਸਮ ਦੀ ਸਲਾਹ।",
    },
    "te": {
        "disease":       "వ్యాధి గుర్తించబడింది: {disease}. తీవ్రత: {severity}.",
        "healthy":       "మీ పంట ఆరోగ్యంగా ఉంది. వ్యాధి లేదు.",
        "actions":       "వెంటనే చేయవలసిన చర్యలు.",
        "step":          "దశ {n}.",
        "treatment":     "సిఫార్సు చేయబడిన చికిత్స.",
        "recovery":      "అంచనా కోలుకునే సమయం: {recovery}",
        "prevention":    "భవిష్యత్తులో వ్యాధి నివారణకు.",
        "weather_intro": "వాతావరణ సలహా.",
    },
}


def _try_gtts(text: str, lang_code: str) -> tuple:
    """
    Generates gTTS audio with language fallbacks for less-supported languages.
    - Punjabi ('pa'): tries 'pa', falls back to 'hi'
    - Odia ('or'): tries 'or', falls back to 'en'
    - Others: tries the given lang_code, falls back to 'en'

    Returns (io.BytesIO with mp3 bytes, effective_lang_code) or (None, None) on total failure.
    """
    fallback_chains = {
        "pa": ["pa", "hi"],
        "or": ["or", "en"],
    }
    codes = fallback_chains.get(lang_code, [lang_code, "en"])
    for code in codes:
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang=code, slow=False).write_to_fp(buf)
            buf.seek(0)
            return buf, code
        except Exception:
            continue
    return None, None


def _flatten_segments(segments: list) -> list:
    """
    Further splits each segment on sentence boundaries ('. ', '.\n', '! ', '? ')
    so every individual sentence becomes its own audio clip with a pause after it.
    Preserves ending punctuation on each piece.
    """
    result = []
    for seg in segments:
        # Split after terminal punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', seg.strip())
        result.extend(p.strip() for p in parts if p.strip())
    return result


def _build_voice_segments(recommendation: dict, disease_display: str,
                          severity: str, lang_code: str) -> list:
    """
    Assembles a list of spoken text segments from the structured recommendation dict.
    Each item in the returned list becomes a separately-paced audio segment
    (a 0.5 s silence is inserted between each when pydub is available).

    Uses VOICE_FRAMING[lang_code] for all framing phrases so the entire
    audio stays in the same language. Recommendation content values come
    from the LLM which was also instructed to write in that language.

    Args:
        recommendation: dict from llm.py
        disease_display: human-readable disease name
        severity: severity string or None
        lang_code: gTTS language code, e.g. "hi"

    Returns:
        List of plain text strings for gTTS to speak.
    """
    f = VOICE_FRAMING.get(lang_code, VOICE_FRAMING["en"])
    segments = []

    # Opening sentence
    if severity and severity.lower() not in ("none", "null", ""):
        segments.append(f["disease"].format(disease=disease_display, severity=severity))
    else:
        segments.append(f["healthy"])

    # Immediate actions
    actions = recommendation.get("immediate_actions", [])
    if actions:
        segments.append(f["actions"])
        for i, action in enumerate(actions, 1):
            segments.append(f["step"].format(n=i) + " " + action)

    # Treatment
    treatments = recommendation.get("treatment", [])
    if treatments:
        segments.append(f["treatment"])
        for t in treatments:
            segments.append(t)

    # Recovery time
    recovery = recommendation.get("recovery_time", "")
    if recovery:
        segments.append(f["recovery"].format(recovery=recovery))

    # Preventive measures
    measures = recommendation.get("preventive_measures", [])
    if measures:
        segments.append(f["prevention"])
        for m in measures:
            segments.append(m)

    return segments


def _generate_with_pydub(segments: list, lang_code: str, output_path: str) -> tuple:
    """
    Generates a single MP3 by concatenating per-segment gTTS audio with
    0.5 second silences between each segment.

    Args:
        segments: list of text strings to speak
        lang_code: gTTS language code
        output_path: path to write the final .mp3

    Returns:
        (True, output_path) on success
        (False, error_message) on failure
    """
    silence = AudioSegment.silent(duration=600)  # 600ms pause between sentences
    combined = None

    for segment in segments:
        if not segment.strip():
            continue
        buf, _ = _try_gtts(segment, lang_code)
        if buf is None:
            continue
        seg_audio = AudioSegment.from_mp3(buf)
        combined = seg_audio if combined is None else combined + silence + seg_audio

    if combined is None:
        return False, "No audio segments to generate"

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    combined.export(output_path, format="mp3")
    return True, output_path


def generate_voice(recommendation: dict, disease_display: str, severity: str,
                   language: str, output_path: str,
                   weather_advice: str = None) -> tuple[bool, str]:
    """
    Generates an MP3 audio file from the agronomic recommendation plus
    optional weather advice.

    Weather advice (already translated by weather.py) is appended AFTER
    the disease recommendation, preceded by the language-specific intro phrase.

    Uses pydub to insert 0.5 s silence between each spoken segment for
    natural pacing. Falls back to a single-pass gTTS call if pydub or
    ffmpeg is not installed.

    Args:
        recommendation: structured dict from llm.py (content already in target language)
        disease_display: human-readable disease name
        severity: severity level string
        language: UI language name e.g. "Kannada"
        output_path: full path where the .mp3 should be saved
        weather_advice: translated spray timing advice to append (optional)

    Returns:
        (True, output_path) on success
        (False, error_message) on failure
    """
    lang_code = LANGUAGE_CODES.get(language, "en")

    # Safety check: fall back to English if lang_code not supported by gTTS
    if lang_code not in GTTS_SUPPORTED:
        lang_code = "en"

    segments = _build_voice_segments(recommendation, disease_display, severity, lang_code)

    # Append translated weather advice after the disease recommendation
    if weather_advice and weather_advice.strip():
        f = VOICE_FRAMING.get(lang_code, VOICE_FRAMING["en"])
        segments.append(f.get("weather_intro", "Weather advisory."))
        segments.append(weather_advice)

    # Sentence-level split: each sentence becomes its own audio clip with a pause
    segments = _flatten_segments(segments)

    # Preferred path: pydub with 600ms silence between sentences
    if PYDUB_AVAILABLE:
        try:
            return _generate_with_pydub(segments, lang_code, output_path)
        except Exception as e:
            print(f"[voice] pydub failed ({e}), falling back to simple gTTS.")

    # Fallback: single gTTS call with language fallbacks
    full_text = " ".join(s for s in segments if s.strip())
    try:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        buf, effective_lang = _try_gtts(full_text, lang_code)
        if buf is None:
            return False, "Voice generation failed: no supported language found"
        with open(output_path, "wb") as f:
            f.write(buf.read())
        return True, output_path
    except Exception as e:
        return False, f"Voice generation failed: {str(e)}"
