# severity.py
# Maps each of the 18 disease classes to a severity level and traffic light color.
# Healthy classes always get None severity and green color.

# Severity levels: None, Low, Moderate, High
# Traffic light colors: green, yellow, red

SEVERITY_MAP = {
    # --- Apple Diseases ---
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Apple Scab"
    },
    "Apple___Black_rot": {
        "severity": "High",
        "color": "red",
        "display": "Apple Black Rot"
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Cedar Apple Rust"
    },
    "Apple___healthy": {
        "severity": None,
        "color": "green",
        "display": "Apple Healthy"
    },

    # --- Grape Diseases ---
    "Grape___Black_rot": {
        "severity": "High",
        "color": "red",
        "display": "Grape Black Rot"
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "High",
        "color": "red",
        "display": "Grape Esca (Black Measles)"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Grape Leaf Blight"
    },
    "Grape___healthy": {
        "severity": None,
        "color": "green",
        "display": "Grape Healthy"
    },

    # --- Tomato Diseases ---
    "Tomato___Bacterial_spot": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Tomato Bacterial Spot"
    },
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Tomato Early Blight"
    },
    "Tomato___Late_blight": {
        "severity": "High",
        "color": "red",
        "display": "Tomato Late Blight"
    },
    "Tomato___Leaf_Mold": {
        "severity": "Low",
        "color": "yellow",
        "display": "Tomato Leaf Mold"
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Tomato Septoria Leaf Spot"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Tomato Spider Mites"
    },
    "Tomato___Target_Spot": {
        "severity": "Moderate",
        "color": "yellow",
        "display": "Tomato Target Spot"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "High",
        "color": "red",
        "display": "Tomato Yellow Leaf Curl Virus"
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "High",
        "color": "red",
        "display": "Tomato Mosaic Virus"
    },
    "Tomato___healthy": {
        "severity": None,
        "color": "green",
        "display": "Tomato Healthy"
    },
}


def get_severity(disease_class: str) -> dict:
    """
    Given a disease class name (e.g. "Tomato___Late_blight"),
    returns its severity level, traffic light color, and human-readable display name.

    Returns a fallback dict if the class is not found.
    """
    if disease_class in SEVERITY_MAP:
        return SEVERITY_MAP[disease_class]

    # Fallback for unknown classes — shouldn't happen with a well-trained model
    return {
        "severity": "Unknown",
        "color": "yellow",
        "display": disease_class.replace("___", " ").replace("_", " ")
    }
