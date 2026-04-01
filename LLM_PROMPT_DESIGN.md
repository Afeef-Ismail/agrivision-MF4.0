# LLM Prompt Design — AgriVision

## 1. Overview

AgriVision uses a **structured prompt engineering** approach to generate actionable agronomic recommendations from an LLM. The LLM is called after the CNN model has already identified the disease and severity — it acts as an **expert agronomist advisor**, not a diagnostician.

**LLM Provider:** Groq (free tier)
**Model:** `llama-3.1-8b-instant`
**Temperature:** 0.3 (low — favours consistent, factual agricultural advice over creative responses)
**Max Tokens:** 1024

---

## 2. Prompt Engineering Approach

### Design Principles

1. **Role Assignment** — The prompt opens by assigning the LLM the role of an "expert agronomist advising an Indian farmer." This grounds the response in domain-specific knowledge and regional context.

2. **Structured Context Injection** — All relevant variables from the prediction pipeline are injected into the prompt as labelled fields, giving the LLM full situational awareness before it generates advice.

3. **Strict Output Schema** — The LLM is instructed to respond **only** with valid JSON matching an exact schema. This eliminates free-text parsing issues and allows the frontend to render each section independently.

4. **No Preamble Rule** — The prompt explicitly states "no preamble, no explanation, no markdown, no extra text" to prevent the LLM from wrapping its response in conversational filler or markdown code fences.

5. **Mode-Conditioned Instructions** — The treatment philosophy (organic vs chemical) is injected as a conditional instruction, changing the LLM's recommendations without altering the prompt structure.

6. **Multi-Language Support** — When the farmer selects a non-English language, an additional instruction block tells the LLM to write all JSON **values** in the target language while keeping JSON **keys** in English. This allows the voice engine (gTTS) to read the response naturally in the farmer's language.

---

## 3. Context Variables

The following variables are injected into the prompt from the prediction pipeline:

| Variable | Source | Example Value | Purpose |
|----------|--------|---------------|---------|
| `crop` | Extracted from model class name | `"Tomato"` | Tells the LLM which crop to advise on |
| `disease` | Severity map display name | `"Tomato Late Blight"` | The detected disease for treatment recommendations |
| `severity` | Severity map level | `"High"` | Helps LLM calibrate urgency of advice |
| `location` | User input (frontend) | `"Mangalore, Karnataka"` | Regional context for India-specific advice |
| `time_of_day` | System clock at inference time | `"morning"` | Context-aware advice (e.g., "spray early morning") |
| `mode` | User toggle (frontend) | `"organic"` or `"chemical"` | Controls treatment philosophy |
| `language` | User dropdown (frontend) | `"Kannada"` | Instructs LLM to write values in target language |

---

## 4. Prompt Template

```
You are an expert agronomist advising an Indian farmer.

Crop: {crop}
Disease Detected: {disease}
Severity: {severity}
Farm Location: {location}
Time of Day: {time_of_day}
Treatment Mode: {mode_instruction}

Provide a complete disease management plan. Respond ONLY with valid JSON —
no preamble, no explanation, no markdown, no extra text.

Return exactly this JSON structure:
{
  "immediate_actions": ["step 1", "step 2", "step 3"],
  "treatment": ["treatment 1", "treatment 2", "treatment 3"],
  "recovery_time": "estimated recovery period as a string",
  "preventive_measures": ["measure 1", "measure 2", "measure 3"],
  "neighbouring_crop_risk": "one sentence about risk to nearby crops"
}

Keep each item concise and practical. Tailor advice to Indian farming conditions.
```

### Mode Instruction (injected dynamically)

- **Organic mode:**
  `"Use only ORGANIC treatments (neem oil, copper fungicide, compost teas, biological controls)."`

- **Chemical mode:**
  `"You may suggest CHEMICAL treatments (fungicides, pesticides, herbicides) as needed."`

### Language Instruction (appended when language ≠ English)

```
IMPORTANT: Write ALL text values inside the JSON in {language_name}.
Every string in the arrays and every string value must be in {language_name}.
The JSON keys must stay in English. Only the values must be translated.
```

---

## 5. Sample Prompt-Response Pairs

### Sample 1: Tomato Late Blight — Chemical Mode — English

**Assembled Prompt:**
```
You are an expert agronomist advising an Indian farmer.

Crop: Tomato
Disease Detected: Tomato Late Blight
Severity: High
Farm Location: Mangalore, Karnataka
Time of Day: morning
Treatment Mode: You may suggest CHEMICAL treatments (fungicides, pesticides, herbicides) as needed.

Provide a complete disease management plan. Respond ONLY with valid JSON — no preamble, no explanation, no markdown, no extra text.

Return exactly this JSON structure:
{
  "immediate_actions": ["step 1", "step 2", "step 3"],
  "treatment": ["treatment 1", "treatment 2", "treatment 3"],
  "recovery_time": "estimated recovery period as a string",
  "preventive_measures": ["measure 1", "measure 2", "measure 3"],
  "neighbouring_crop_risk": "one sentence about risk to nearby crops"
}

Keep each item concise and practical. Tailor advice to Indian farming conditions.
```

**LLM Response:**
```json
{
  "immediate_actions": [
    "Remove and destroy all visibly infected leaves and stems immediately.",
    "Isolate affected tomato plants from healthy ones to prevent spore spread.",
    "Avoid overhead irrigation — switch to drip irrigation to reduce leaf wetness."
  ],
  "treatment": [
    "Apply Mancozeb 75% WP at 2.5 g/L of water as a protective fungicide spray.",
    "Follow up with Metalaxyl + Mancozeb (Ridomil Gold) at 2 g/L for systemic control.",
    "Repeat spray every 7–10 days during wet weather conditions."
  ],
  "recovery_time": "3–4 weeks with consistent fungicide application and dry weather.",
  "preventive_measures": [
    "Use certified Late Blight-resistant tomato varieties like Arka Rakshak.",
    "Practice 2-year crop rotation — avoid Solanaceae crops in the same plot.",
    "Maintain proper plant spacing (60 cm) for air circulation."
  ],
  "neighbouring_crop_risk": "Late Blight (Phytophthora infestans) can rapidly spread to nearby potato fields — alert neighbouring farmers immediately."
}
```

---

### Sample 2: Apple Scab — Organic Mode — Hindi

**Assembled Prompt:**
```
You are an expert agronomist advising an Indian farmer.

Crop: Apple
Disease Detected: Apple Scab
Severity: Moderate
Farm Location: Shimla, Himachal Pradesh
Time of Day: afternoon
Treatment Mode: Use only ORGANIC treatments (neem oil, copper fungicide, compost teas, biological controls).

Provide a complete disease management plan. Respond ONLY with valid JSON — no preamble, no explanation, no markdown, no extra text.

Return exactly this JSON structure:
{
  "immediate_actions": ["step 1", "step 2", "step 3"],
  "treatment": ["treatment 1", "treatment 2", "treatment 3"],
  "recovery_time": "estimated recovery period as a string",
  "preventive_measures": ["measure 1", "measure 2", "measure 3"],
  "neighbouring_crop_risk": "one sentence about risk to nearby crops"
}

Keep each item concise and practical. Tailor advice to Indian farming conditions.

IMPORTANT: Write ALL text values inside the JSON in Hindi (हिंदी). Every string in the arrays and every string value must be in Hindi (हिंदी). The JSON keys must stay in English. Only the values must be translated.
```

**LLM Response:**
```json
{
  "immediate_actions": [
    "संक्रमित पत्तियों और फलों को तुरंत तोड़कर नष्ट करें।",
    "पेड़ के नीचे गिरी हुई पत्तियों को इकट्ठा करके जला दें।",
    "प्रभावित शाखाओं की छंटाई करें ताकि हवा का प्रवाह बेहतर हो सके।"
  ],
  "treatment": [
    "बोर्डो मिश्रण (1%) का छिड़काव करें — यह एक जैविक तांबा फफूंदनाशक है।",
    "नीम तेल (5 मिली/लीटर पानी) का हर 10 दिन में छिड़काव करें।",
    "ट्राइकोडर्मा विरिडी का मिट्टी और पत्तियों पर प्रयोग करें।"
  ],
  "recovery_time": "उचित उपचार के साथ 3–5 सप्ताह में सुधार होगा।",
  "preventive_measures": [
    "स्कैब-प्रतिरोधी सेब की किस्में जैसे फ्लोरिना लगाएं।",
    "सर्दियों में निष्क्रिय मौसम में तांबा फफूंदनाशक का छिड़काव करें।",
    "पेड़ों के बीच पर्याप्त दूरी रखें और छंटाई नियमित करें।"
  ],
  "neighbouring_crop_risk": "सेब के स्कैब रोग से आसपास के सेब और नाशपाती के पेड़ प्रभावित हो सकते हैं — गिरी हुई पत्तियों को साफ रखें।"
}
```

---

### Sample 3: Grape Healthy — Chemical Mode — English

**Assembled Prompt:**
```
You are an expert agronomist advising an Indian farmer.

Crop: Grape
Disease Detected: Grape Healthy
Severity: None (Healthy plant)
Farm Location: Nashik, Maharashtra
Time of Day: evening
Treatment Mode: You may suggest CHEMICAL treatments (fungicides, pesticides, herbicides) as needed.

Provide a complete disease management plan. Respond ONLY with valid JSON — no preamble, no explanation, no markdown, no extra text.

Return exactly this JSON structure:
{
  "immediate_actions": ["step 1", "step 2", "step 3"],
  "treatment": ["treatment 1", "treatment 2", "treatment 3"],
  "recovery_time": "estimated recovery period as a string",
  "preventive_measures": ["measure 1", "measure 2", "measure 3"],
  "neighbouring_crop_risk": "one sentence about risk to nearby crops"
}

Keep each item concise and practical. Tailor advice to Indian farming conditions.
```

**LLM Response:**
```json
{
  "immediate_actions": [
    "No immediate action required — your grape vine is healthy.",
    "Continue regular monitoring of leaves for early signs of downy mildew or black rot.",
    "Maintain current irrigation and fertilisation schedule."
  ],
  "treatment": [
    "No treatment needed at this time.",
    "Apply a preventive copper-based spray before monsoon season as a precaution.",
    "Ensure canopy management is up to date to promote airflow."
  ],
  "recovery_time": "Not applicable — plant is healthy.",
  "preventive_measures": [
    "Prune vines regularly to maintain open canopy and reduce humidity.",
    "Apply balanced NPK fertiliser (10:26:26) during flowering stage.",
    "Install pheromone traps to monitor pest activity in the vineyard."
  ],
  "neighbouring_crop_risk": "No disease risk to neighbouring crops at this time."
}
```

---

## 6. Response Parsing & Fallback Strategy

### Parsing (`llm.py → _parse_llm_response`)
1. Strip any markdown code fences (` ```json `) that the LLM might add despite instructions
2. Parse the cleaned string as JSON
3. Return the parsed dict directly to the pipeline

### Fallback (if Groq fails)
If the Groq API is unavailable (network error, rate limit, missing key), the system returns a **safe static fallback** dictionary with generic but useful agricultural advice. The app never crashes — the farmer always gets some recommendation.

```json
{
  "immediate_actions": [
    "Remove and destroy visibly infected plant parts.",
    "Isolate affected plants from healthy ones.",
    "Avoid overhead irrigation to reduce moisture spread."
  ],
  "treatment": [
    "Apply a broad-spectrum fungicide or bactericide as appropriate.",
    "Consult your local Krishi Vigyan Kendra (KVK) for approved products."
  ],
  "recovery_time": "2–4 weeks with proper treatment.",
  "preventive_measures": [
    "Practice crop rotation every season.",
    "Use certified disease-resistant seed varieties.",
    "Maintain proper plant spacing for airflow."
  ],
  "neighbouring_crop_risk": "Monitor neighbouring crops for similar symptoms and take preventive action."
}
```

---

## 7. Why This Approach Works

| Design Choice | Rationale |
|---------------|-----------|
| Low temperature (0.3) | Agricultural advice must be factual and consistent, not creative |
| Strict JSON schema | Enables structured frontend rendering and voice generation |
| Role assignment | Grounds the LLM in agronomic expertise rather than general knowledge |
| Context injection | Disease + severity + location + time + mode = highly relevant advice |
| No-preamble rule | Prevents parsing failures from conversational filler text |
| Language instruction | Allows gTTS to speak advice naturally in the farmer's native language |
| Static fallback | Ensures the app is useful even without internet connectivity |
