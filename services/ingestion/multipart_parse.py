"""Parse and normalize values coming from multipart form fields."""


def parse_extract_images(raw: str) -> bool:
    x = raw.strip().lower()
    if x in ("true", "1", "yes", "on"):
        return True
    if x in ("false", "0", "no", "off", ""):
        return False
    raise ValueError("extract_images must be true or false")


def normalize_job_id(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    return s or None
