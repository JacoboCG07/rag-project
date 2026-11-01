"""
Tagged multi-value metadata fields (pages, chapters) stored as VARCHAR in Milvus.

Each value is wrapped in asterisks and separated by spaces:

    ["VI", "VII"]  ->  "*VI* *VII*"
    5              ->  "*5*"

The asterisks delimit every value on both sides, so a single infix LIKE matches a value
wherever it appears without partial matches (searching "V" never matches "VI" or "XV"):

    chapters like "%*VII*%"

Infix LIKE patterns require Milvus >= 2.4.
"""

import re
from typing import Any, Iterable, List

_TAG = "*"
_TAGGED_RE = re.compile(r"\*([^*]+)\*")
# Characters with special meaning in the tag format or in a Milvus string literal / LIKE pattern
_UNSAFE_CHARS = re.compile(r'[*"%\\]')


def is_tagged(value: Any) -> bool:
    """True if the value is already a string in tagged format ("*a* *b*")."""
    return isinstance(value, str) and value.strip().startswith(_TAG)


def to_tagged(value: Any) -> str:
    """
    Serialize one or several values to the tagged format.

    Args:
        value: list/tuple/set of values, a single value (str/int), None or False.

    Returns:
        str: "*a* *b*" for lists, "*a*" for a single value, "" for empty values (None, False, "", []).
            Strings that are already tagged are returned unchanged.
    """
    if value is None or value is False or value == "" or value == []:
        return ""
    if is_tagged(value):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = [str(value).strip()]
    return " ".join(f"{_TAG}{item}{_TAG}" for item in items if item)


def parse_tagged(value: Any) -> List[str]:
    """
    Parse a stored value back into a list.

    Supports the tagged format and the legacy comma-separated format ("VI,VII").
    """
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if is_tagged(text):
        return [v.strip() for v in _TAGGED_RE.findall(text) if v.strip()]
    return [v.strip() for v in text.split(",") if v.strip()]


def like_any(field: str, values: Iterable[Any]) -> str:
    """
    Build a Milvus expression matching rows whose tagged `field` contains any of `values`.

    Example:
        like_any("chapters", ["VII", "VIII"])
        -> '(chapters like "%*VII*%" or chapters like "%*VIII*%")'

    Returns:
        str: The expression, or "" if there are no usable values.
    """
    clean = []
    for v in values or []:
        s = _UNSAFE_CHARS.sub("", str(v)).strip()
        if s and s not in clean:
            clean.append(s)
    if not clean:
        return ""
    parts = [f'{field} like "%{_TAG}{v}{_TAG}%"' for v in clean]
    return parts[0] if len(parts) == 1 else "(" + " or ".join(parts) + ")"
