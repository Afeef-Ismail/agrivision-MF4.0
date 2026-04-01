# disease_info.py
# Contains factual information about which neighbouring crops are at risk
# from each of the 18 disease classes.
# Used to show a warning banner to farmers about potential spread.

# Each entry maps a disease class to a warning string about neighbouring crop risks.
# Healthy classes have no risk — returns an empty string.

NEIGHBOURING_RISK_MAP = {
    # --- Apple ---
    "Apple___Apple_scab": (
        "Apple Scab (Venturia inaequalis) can spread to other apple trees and "
        "pear trees nearby. Ensure spacing and remove fallen infected leaves."
    ),
    "Apple___Black_rot": (
        "Apple Black Rot can spread to other apple and pear trees through infected "
        "wood and fruit. Remove and destroy mummified fruits immediately."
    ),
    "Apple___Cedar_apple_rust": (
        "Cedar Apple Rust requires both juniper/cedar and apple as hosts. "
        "Remove nearby juniper or cedar trees if possible to break the disease cycle."
    ),
    "Apple___healthy": "",

    # --- Grape ---
    "Grape___Black_rot": (
        "Grape Black Rot spreads rapidly through a vineyard via rain splash. "
        "It can infect all Vitis species nearby. Remove mummified berries to reduce spore load."
    ),
    "Grape___Esca_(Black_Measles)": (
        "Grape Esca is a wood disease that spreads slowly but can devastate the entire vineyard. "
        "Infected vines can transmit via pruning tools — disinfect tools between plants."
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape Leaf Blight can spread to adjacent vines through wind and rain. "
        "Maintain good air circulation by proper canopy management."
    ),
    "Grape___healthy": "",

    # --- Tomato ---
    "Tomato___Bacterial_spot": (
        "Tomato Bacterial Spot (Xanthomonas) can spread to pepper plants in nearby plots. "
        "Avoid overhead irrigation and working in wet fields."
    ),
    "Tomato___Early_blight": (
        "Tomato Early Blight can spread to potato, eggplant, and pepper crops nearby. "
        "It is soilborne and airborne — rotate crops and remove infected debris."
    ),
    "Tomato___Late_blight": (
        "WARNING: Tomato Late Blight (Phytophthora infestans) is highly contagious. "
        "It can devastate nearby potato fields within days under cool, wet conditions. "
        "Alert neighbouring farmers immediately and apply fungicide as a preventive measure."
    ),
    "Tomato___Leaf_Mold": (
        "Tomato Leaf Mold thrives in high humidity greenhouses and can spread to other "
        "tomato plants via air currents. Improve ventilation urgently."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Tomato Septoria Leaf Spot spreads via rain splash and tools. "
        "It can also infect eggplant. Remove lower infected leaves to slow spread."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Spider mites can spread to cucumber, bean, eggplant, and strawberry crops nearby. "
        "They move on wind and clothing — avoid moving through infected plots."
    ),
    "Tomato___Target_Spot": (
        "Tomato Target Spot can spread to pepper and eggplant. "
        "Reduce humidity and avoid overhead watering to limit spread."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "ALERT: Tomato Yellow Leaf Curl Virus is spread by whiteflies. "
        "Nearby pepper, bean, and cucumber crops are at risk. "
        "Control whitefly populations immediately using sticky traps or insecticides."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato Mosaic Virus spreads by contact — through hands, tools, and infected seeds. "
        "It can infect pepper, eggplant, and tobacco. Wash hands and disinfect tools constantly."
    ),
    "Tomato___healthy": "",
}


def get_neighbouring_risk(disease_class: str) -> str:
    """
    Given a disease class name, returns a warning string about neighbouring crop risks.
    Returns an empty string for healthy plants or unknown classes.
    """
    return NEIGHBOURING_RISK_MAP.get(disease_class, "")
