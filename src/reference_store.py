"""Versioned storage for official reference lists (IPOS / MyIPO).

Every confirmed upload is saved as a new timestamped version under
data/reference/<authority>/versions/, and data/reference/<authority>/current.json
is updated to point at it. Older versions are kept so you can roll back if a
new official list turns out to be wrong, and so you always know which list
version a past check report was run against.
"""
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
AUTHORITIES = ["ipos", "myipo"]


def _authority_dir(authority):
    d = DATA_DIR / authority
    (d / "versions").mkdir(parents=True, exist_ok=True)
    return d


def save_new_version(authority, rows, source_filename):
    """rows: list of {"class": str, "term": str}. Returns the version id."""
    authority = authority.lower()
    if authority not in AUTHORITIES:
        raise ValueError(f"Unknown authority: {authority}")

    d = _authority_dir(authority)
    version_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "authority": authority,
        "version_id": version_id,
        "source_filename": source_filename,
        "term_count": len(rows),
        "rows": rows,
    }
    version_path = d / "versions" / f"{version_id}.json"
    version_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pointer_path = d / "current.json"
    pointer_path.write_text(json.dumps({"current_version": version_id}, indent=2), encoding="utf-8")

    return version_id


def get_current_version_id(authority):
    pointer_path = DATA_DIR / authority / "current.json"
    if not pointer_path.exists():
        return None
    return json.loads(pointer_path.read_text(encoding="utf-8"))["current_version"]


def load_current(authority):
    version_id = get_current_version_id(authority)
    if version_id is None:
        return None
    return load_version(authority, version_id)


def load_version(authority, version_id):
    path = DATA_DIR / authority / "versions" / f"{version_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_versions(authority):
    d = DATA_DIR / authority / "versions"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append({
            "version_id": data["version_id"],
            "source_filename": data["source_filename"],
            "term_count": data["term_count"],
        })
    return out


def status_summary():
    summary = {}
    for authority in AUTHORITIES:
        current_id = get_current_version_id(authority)
        current = load_version(authority, current_id) if current_id else None
        summary[authority] = {
            "current_version": current_id,
            "term_count": current["term_count"] if current else 0,
            "source_filename": current["source_filename"] if current else None,
        }
    return summary
