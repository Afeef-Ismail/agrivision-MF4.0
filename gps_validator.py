# gps_validator.py
# Reads GPS EXIF metadata from drone/phone images and validates that the
# image was taken within a known Indian farming region.
#
# Uses only Pillow (PIL) — no external API, no internet required.
# GPS EXIF data is embedded by most modern smartphones and drone cameras.
# Google/stock images typically have no GPS data — this is expected and fine.
#
# Three outcomes:
#   - No GPS in image → return None  (no warning — don't penalise non-GPS images)
#   - GPS found, inside a known region → return ✅ confirmation string
#   - GPS found, outside all regions → return ⚠️ warning string

from PIL import Image, ExifTags

# Bounding boxes for known Indian agricultural regions.
# Format: {"lat": (min_lat, max_lat), "lon": (min_lon, max_lon)}
FARM_REGIONS = {
    "Punjab/Haryana":    {"lat": (28.5, 32.5), "lon": (73.5, 77.5)},
    "Maharashtra":       {"lat": (15.5, 22.5), "lon": (72.5, 80.5)},
    "Karnataka":         {"lat": (11.5, 18.5), "lon": (74.0, 78.5)},
    "Kerala":            {"lat": (8.0,  12.5), "lon": (74.5, 77.5)},
    "Tamil Nadu":        {"lat": (8.0,  13.5), "lon": (76.5, 80.5)},
    "Andhra Pradesh":    {"lat": (12.5, 19.5), "lon": (76.5, 84.5)},
    "Gujarat":           {"lat": (20.0, 24.5), "lon": (68.0, 74.5)},
    "West Bengal":       {"lat": (21.5, 27.5), "lon": (85.5, 89.5)},
    "Uttar Pradesh":     {"lat": (23.5, 30.5), "lon": (77.0, 84.5)},
    "Rajasthan":         {"lat": (23.0, 30.5), "lon": (69.5, 78.5)},
}


def _dms_to_decimal(dms, ref: str) -> float:
    """
    Converts GPS EXIF DMS (degrees, minutes, seconds) to decimal degrees.

    EXIF stores GPS as a tuple of three IFDRational values:
        (degrees, minutes, seconds)
    The ref string ("N"/"S" for latitude, "E"/"W" for longitude) determines sign.
    """
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(image_path: str):
    """
    Reads GPS coordinates from an image's EXIF metadata using Pillow.

    Args:
        image_path: path to the image file

    Returns:
        (latitude, longitude) as floats if GPS data is present,
        None if no GPS data found or if the image has no EXIF at all.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None

        # Build a tag-name → value dict for easy lookup
        exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None

        # Map GPS sub-tag IDs to names
        gps = {ExifTags.GPSTAGS.get(key, key): value for key, value in gps_info.items()}

        lat_dms = gps.get("GPSLatitude")
        lat_ref = gps.get("GPSLatitudeRef")
        lon_dms = gps.get("GPSLongitude")
        lon_ref = gps.get("GPSLongitudeRef")

        if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
            return None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
        return lat, lon

    except Exception:
        # Any error (corrupt EXIF, missing tags, etc.) → treat as no GPS
        return None


def validate_farm_region(lat: float, lon: float) -> str | None:
    """
    Checks whether the given coordinates fall within any known Indian farming region.

    Args:
        lat: decimal latitude
        lon: decimal longitude

    Returns:
        Region name string if matched, None if outside all known regions.
    """
    for region, bounds in FARM_REGIONS.items():
        if (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                bounds["lon"][0] <= lon <= bounds["lon"][1]):
            return region
    return None


def get_gps_warning(image_path: str) -> str | None:
    """
    Extracts GPS from the image and validates it against known Indian farming regions.

    Args:
        image_path: path to the uploaded image file

    Returns:
        None — if no GPS data in image (expected for most Google images; no penalty)
        "✅ GPS: Image captured in [region]" — GPS confirmed in a known farm region
        "⚠️ GPS Warning: ..." — GPS found but outside all known Indian farm regions
    """
    coords = extract_gps(image_path)
    if coords is None:
        return None

    lat, lon = coords
    region = validate_farm_region(lat, lon)

    if region:
        return f"✅ GPS: Image captured in {region}"

    return (
        "⚠️ GPS Warning: Image location does not match any known Indian farm region. "
        "Please verify this image is from the correct field."
    )
