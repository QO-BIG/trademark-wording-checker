"""Free, zero-cost local matching against the official IPOS/MyIPO wording
lists. Whether an item's wording exactly matches a real official term is a
yes/no fact — look it up, don't ask an LLM. Only items with no exact match
("non-standard" candidates) need to go to Claude for a suggestion.

Lookups are O(1) dict access keyed on (normalised class, normalised wording),
so this is instant even against MyIPO's ~136k-term list.
"""
import json
import os
import re

# data/reference/, resolved relative to this file so it works regardless of
# the process's current working directory (matters on Streamlit Cloud).
DEFAULT_REFERENCE_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference"
)

_DASH_MAP = str.maketrans({
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",  # left single smart quote
    "’": "'",  # right single smart quote
    "“": '"',  # left double smart quote
    "”": '"',  # right double smart quote
})


def normalize_term(text):
    """Lowercase, collapse whitespace, standardise quotes/dashes, strip a
    single trailing '.', ';', ':', or ',' — and nothing else. Internal
    commas are preserved on purpose: real official terms routinely contain
    them (e.g. "retail services for pharmaceutical, veterinary and sanitary
    preparations and medical supplies"), so splitting or normalising on
    commas would corrupt genuine terms. Any other difference (word choice,
    singular/plural, spelling) is left alone, because MyIPO in particular
    marks any such difference as non-standard — this function's job is
    exact-match detection, not fuzzy similarity.
    """
    text = (text or "").translate(_DASH_MAP)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.rstrip(".;:,")


def normalize_class(cls):
    """Extracts the class number and strips leading zeros/labels, so "09",
    "9", and "Class 9" all normalise to "9". Returns "" if no digits found."""
    digits = re.sub(r"\D", "", cls or "")
    return digits.lstrip("0") or ("0" if digits else "")


def split_on_semicolons(term):
    """Splits a single wording string into sub-items on ';' only — never on
    ',', since official terms routinely contain internal commas that a
    comma-split would wrongly tear apart. Returns [term] unchanged if there's
    nothing to split."""
    parts = [p.strip() for p in (term or "").split(";")]
    return [p for p in parts if p] or [term]


def load_reference_rows(path):
    """Loads one authority's versioned reference JSON (the
    {"rows": [{"class", "term"}, ...]} shape written by the old
    reference_store.py / restored from git history) and returns the row list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["rows"]


def load_authority_reference(authority, base_dir=None):
    """Resolves data/reference/{authority}/current.json -> its
    current_version -> the matching versions/{version}.json, and returns
    that version's rows. Raises if the authority's reference data isn't
    present — callers should treat that as "local matching unavailable"
    and fall back to sending everything to the agent, not as a hard crash."""
    base_dir = base_dir or DEFAULT_REFERENCE_BASE_DIR
    auth_dir = os.path.join(base_dir, (authority or "").strip().lower())
    with open(os.path.join(auth_dir, "current.json"), encoding="utf-8") as f:
        current = json.load(f)
    version_path = os.path.join(auth_dir, "versions", f"{current['current_version']}.json")
    return load_reference_rows(version_path)


class ReferenceIndex:
    """Pre-built lookup index for one authority's official wording list."""

    def __init__(self, rows):
        # (norm_class, norm_term) -> official term, for an exact class+wording match
        self.by_class_and_term = {}
        # norm_term -> list of (official_class_label, official_term), for the
        # "right wording, wrong class" check across every class
        self.by_term = {}

        for row in rows:
            norm_cls = normalize_class(row["class"])
            norm_term = normalize_term(row["term"])
            self.by_class_and_term[(norm_cls, norm_term)] = row["term"]
            self.by_term.setdefault(norm_term, []).append((row["class"], row["term"]))

    def lookup(self, cls, term):
        """Returns one of:
        - ("standard", official_term, None) — exact (class, wording) match
        - ("standard_wrong_class", official_term, correct_class) — wording
          matches a real official term, but under a different class
        - ("non_standard", None, None) — no exact match anywhere; needs
          the agent to suggest a correction
        """
        norm_cls = normalize_class(cls)
        norm_term = normalize_term(term)

        official = self.by_class_and_term.get((norm_cls, norm_term))
        if official is not None:
            return "standard", official, None

        other_class_hits = self.by_term.get(norm_term)
        if other_class_hits:
            # Prefer a hit whose class is unambiguous when the wording only
            # exists under one other class; otherwise report the first.
            correct_class, official_term = other_class_hits[0]
            return "standard_wrong_class", official_term, correct_class

        return "non_standard", None, None


def classify_items_locally(items, reference_index):
    """items: list of {"class": ..., "term": ...} (the extraction step's
    output shape). Splits any semicolon-joined multi-term wording into
    separate items first, then classifies each for free against the
    reference list, and flags exact-duplicate wording.

    Returns a list of dicts, one per (possibly split) item:
        {
            "input": original wording text for this sub-item,
            "class": original class as given,
            "status": "standard" | "standard_wrong_class" | "non_standard",
            "official_term": the matching official term, or None,
            "correct_class": set only for "standard_wrong_class",
            "is_duplicate": True if this normalised wording appears more
                than once in the input list,
            "needs_agent": True only for "non_standard" items — this is
                the filter to apply before sending anything to Claude.
        }
    """
    expanded = []
    for item in items:
        cls = item.get("class", "")
        for sub_term in split_on_semicolons(item.get("term", "")):
            expanded.append({"class": cls, "term": sub_term})

    norm_term_counts = {}
    for item in expanded:
        key = normalize_term(item["term"])
        norm_term_counts[key] = norm_term_counts.get(key, 0) + 1

    results = []
    for item in expanded:
        cls, term = item["class"], item["term"]
        status, official_term, correct_class = reference_index.lookup(cls, term)
        results.append({
            "input": term,
            "class": cls,
            "status": status,
            "official_term": official_term,
            "correct_class": correct_class,
            "is_duplicate": norm_term_counts[normalize_term(term)] > 1,
            "needs_agent": status == "non_standard",
        })

    return results
