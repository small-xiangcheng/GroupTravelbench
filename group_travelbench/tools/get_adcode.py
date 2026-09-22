from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

DATA_FILE = Path(__file__).with_name("adcode.txt")
PLACEHOLDER_SUFFIXES = ("市辖区",)


def _load_entries(path: Path) -> List[Tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    entries: List[Tuple[str, str]] = []
    for line_no, line in enumerate(lines[1:], start=2):  # skip header
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Unexpected column count on line {line_no}: {line!r}")

        name, code, _city_code = parts
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid adcode {code!r} on line {line_no}")

        entries.append((name, code))

    return entries


def _is_placeholder(name: str) -> bool:
    return name.endswith(PLACEHOLDER_SUFFIXES)


def build_fullname2adcode(path: Path = DATA_FILE) -> Dict[str, int]:
    """
    Parse the adcode file and build a mapping from full administrative names to adcodes.
    Placeholder nodes such as \"市辖区\" are skipped when composing full names.
    """

    entries = _load_entries(path)
    code_to_name = {code: name for name, code in entries}

    def parent_code(code: str) -> str | None:
        if code.endswith("0000"):
            return None
        if code.endswith("00"):
            parent = f"{code[:2]}0000"
            return parent if parent in code_to_name else None

        candidate = f"{code[:4]}00"
        if candidate in code_to_name:
            return candidate

        fallback = f"{code[:2]}0000"
        return fallback if fallback in code_to_name else None

    @lru_cache(None)
    def fullname(code: str, include_placeholder: bool) -> str:
        name = code_to_name[code]
        parent = parent_code(code)

        parent_full = fullname(parent, False) if parent else ""
        if _is_placeholder(name):
            if not include_placeholder:
                return parent_full

            # Drop duplicated parent name prefix for cleaner full names.
            if parent:
                parent_name = code_to_name[parent]
                if name.startswith(parent_name):
                    name = name[len(parent_name) :]

        return f"{parent_full}{name}" if parent_full else name

    fullname2adcode: Dict[str, int] = {}
    for name, code in entries:
        full_name = fullname(code, True)
        if not full_name:
            continue

        code_value = int(code)
        existing = fullname2adcode.get(full_name)
        if existing is not None and existing != code_value:
            raise ValueError(f"Conflicting adcode for {full_name}: {existing} vs {code_value}")

        fullname2adcode[full_name] = code_value

    return fullname2adcode


fullname2adcode = build_fullname2adcode()


def is_valid_terminal_name(name: str) -> Dict[str, object]:
    """
    Tool name: is_valid_terminal_name
    Description: Check whether the input is a valid single-level
    administrative name (a province, city, or county/district on its own).
    Parameters:
        name (str): Place name to check, e.g. "江西省", "南昌市", "朝阳区".
    Returns:
        dict: {
            'valid': bool,           # whether it is a valid single-level name
            'matches': list,         # matched full administrative records
            'reason': str            # explanation
        }
    Examples:
        - "江西省" / "江西" → valid=True (province alone)
        - "南昌" / "南昌市" → valid=True (city alone)
        - "江西南昌" / "江西省南昌市" → valid=False (compound; pass one level only)
        - "朝阳区" → valid=True (district alone; may match more than one)
    """
    name = name.strip()
    
    if not name:
        return {"valid": False, "matches": [], "reason": "输入为空"}
    
    terminal_suffixes = ("省", "市", "区", "县", "旗", "盟", "州", "地区", "林区", "特区")
    
    # Several administrative suffixes usually mean a compound place name
    suffix_count = sum(1 for s in terminal_suffixes if s in name)
    if suffix_count > 1:
        return {
            "valid": False, 
            "matches": [],
            "reason": "输入包含多级行政区划名称，请只输入单级地名（如'江西省'或'南昌市'，而非'江西省南昌市'）"
        }
    
    # A suffix followed by more text is also a compound place name
    for suffix in terminal_suffixes:
        if suffix in name:
            idx = name.find(suffix)
            if idx != -1 and idx + len(suffix) < len(name):
                remaining = name[idx + len(suffix):]
                return {
                    "valid": False,
                    "matches": [],
                    "reason": f"输入是组合地名（'{name[:idx+len(suffix)]}' + '{remaining}'），请只输入单级地名"
                }
    
    # Find every administrative unit whose full name ends with this token
    matches = [
        {"full_name": full_name, "adcode": code}
        for full_name, code in fullname2adcode.items()
        if full_name.endswith(name)
    ]
    
    # If there is no exact suffix match, try common short names (e.g. "江西" → "江西省")
    if not matches:
        for suffix in terminal_suffixes:
            name_with_suffix = name + suffix
            suffix_matches = [
                {"full_name": full_name, "adcode": code}
                for full_name, code in fullname2adcode.items()
                if full_name.endswith(name_with_suffix)
            ]
            if suffix_matches:
                matches.extend(suffix_matches)
    
    matches.sort(key=lambda item: (len(item["full_name"]), item["full_name"]))
    
    if not matches:
        return {"valid": False, "matches": [], "reason": "未找到匹配的行政区划"}
    
    # A match means this is a valid single-level place name
    return {
        "valid": True,
        "matches": matches,
        "reason": "合法的单级地名" + (f"，共{len(matches)}个匹配" if len(matches) > 1 else "")
    }


def get_adcode(name: str) -> Dict[str, object]:
    """
    Tool name: get_adcode
    Description: Resolve a Chinese administrative name (province, city,
    or county/district) to its 6-digit ADCODE using the loaded dataset.
    Parameters:
        name (str): Administrative name. Accepts a full path
            (e.g. "北京市", "北京市朝阳区") or a leaf name (e.g. "朝阳区").
    Returns:
        dict: {
            'matches': list[{'full_name': str, 'adcode': int}],  # candidates, by name length then lex order
            'note': str                                         # exact hit, ambiguous, or not found
        }
    """

    name = name.strip()

    # Exact match first.
    if name in fullname2adcode:
        code = fullname2adcode[name]
        match = {"full_name": name, "adcode": code}
        return {"resolved": match, "matches": [match], "note": "exact match"}

    # Fallback: suffix matches (e.g., name == '朝阳区' → multiple hits).
    matches = [
        {"full_name": full_name, "adcode": code}
        for full_name, code in fullname2adcode.items()
        if full_name.endswith(name)
    ]
    matches.sort(key=lambda item: (len(item["full_name"]), item["full_name"]))

    resolved = matches[0] if len(matches) == 1 else None
    note = "no match" if not matches else ("ambiguous matches" if not resolved else "suffix match")

    return {"resolved": resolved, "matches": matches, "note": note}


def validate_city_name(city_name: str) -> bool:
    """
    Validate a city / district name, including short forms without a suffix.

    For example, '平凉' is also tried as '平凉市' and other common suffixes.

    Parameters:
        city_name (str): City or district name to validate.
    Returns:
        bool: True if recognized, False otherwise.
    """
    city_name = city_name.strip()
    if not city_name:
        return False

    # Direct match
    adcode_result = get_adcode(city_name)
    if adcode_result.get("matches"):
        return True

    # Try appending common administrative suffixes
    suffixes = ["市", "省", "区", "县", "州", "盟", "旗", "地区"]
    for suffix in suffixes:
        adcode_result = get_adcode(city_name + suffix)
        if adcode_result.get("matches"):
            return True

    return False


__all__ = ["build_fullname2adcode", "fullname2adcode", "get_adcode", "is_valid_terminal_name", "validate_city_name"]


if __name__ == "__main__":
    mapping = fullname2adcode
    print(f"Loaded {len(mapping)} fullname entries from {DATA_FILE.name}")
