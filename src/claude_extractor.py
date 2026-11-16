"""Extracting a goods/services wording list from a client's document, in
whatever format they happened to send it.

Most client files match one of a handful of common layouts (see
extraction_rules.py) and are extracted with plain code, for free — a clean
one-item-per-line list, a numbered table, a bilingual spreadsheet, or an
instruction letter with the goods list embedded as "GOODS IN CLASS 5: item
one; item two" inside a metadata table. Claude is only called when none of
those layouts confidently apply, and even then it's asked for WHERE each
class's text begins and ends (not to retype it) — that keeps the call small
(locations, not a full retyped list — output is ~5x the cost of input, so
this is the single biggest lever) and guarantees the extracted wording is
character-for-character what the client wrote, since the surrounding code
cuts it from the original text rather than Claude reproducing it.
"""
import io
import re

import pandas as pd
import pdfplumber
from docx import Document

from . import extraction_rules
from .claude_client import authority_label, get_client

DEFAULT_MODEL = "claude-haiku-4-5"
# Overall safety ceiling on how much of a document gets sent at all.
MAX_DUMP_CHARS = 200000
# Output here is a handful of {class, start, end} markers, not retyped item
# text, so it doesn't scale with document size the way full extraction did —
# a small cap is genuinely enough, not just optimistic.
LOCATION_MAX_TOKENS = 2048
# If a document has so many distinct class sections that even the location
# list overflows, bisect and retry each half rather than losing the rest
# silently. Guards against runaway recursion on a pathological document.
MAX_SPLIT_DEPTH = 4

LOCATION_TOOL = {
    "name": "report_extraction_locations",
    "description": (
        "Report WHERE each NICE class's goods/services text begins and ends in the document content "
        "below, so the surrounding code can cut and split it out verbatim. Do not retype, correct, or "
        "paraphrase the wording yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {
                            "type": "string",
                            "description": "NICE class number for this section, as stated or clearly implied by the document. Empty string if not determinable.",
                        },
                        "start": {
                            "type": "string",
                            "description": "The first 3-8 words of this class's goods/services text, copied EXACTLY verbatim from the document content below (identical spelling, spacing, case) so the surrounding code can locate where it begins.",
                        },
                        "end": {
                            "type": "string",
                            "description": "The last 3-8 words of this class's goods/services text (inclusive of the final item), copied EXACTLY verbatim from the document content below, so the surrounding code can locate where it ends.",
                        },
                    },
                    "required": ["class", "start", "end"],
                },
            }
        },
        "required": ["sections"],
    },
}

LOCATION_SYSTEM_PROMPT = """You are helping locate the goods/services wording in a client's trademark \
filing document for a {authority} application, so surrounding code can extract it verbatim. Clients send \
this in all kinds of formats — a plain list, a numbered table, an instruction letter with the goods list \
embedded part-way through (e.g. "GOODS IN CLASS 5: item one; item two; item three" inside a metadata table, \
mixed in with applicant details and boilerplate remarks), a bilingual spreadsheet with English and Chinese \
columns side by side, a spreadsheet with a merged "Class" column where the class number is only written \
once per block of rows, etc.

For each distinct NICE class's goods/services text in the document, report ONLY where it starts and ends \
(a short verbatim snippet from each end) — do not type out the items yourself. Rules:
- The start/end snippets must be copied EXACTLY as written in the document below (same spelling, spacing, \
case) — not corrected, standardized, or paraphrased. Surrounding code does exact text matching against them.
- If the document has multiple sheets/sections for different designated countries, only report the section \
relevant to {authority} if you can tell which is which.
- Ignore anything that is not a goods/services item: applicant name/address/dates, headers, column titles, \
page numbers, running row numbers, filing instructions, disclaimers, trademark names, and any Chinese or \
other-language translation column or text.
- If a class's goods/services text is not contiguous (e.g. split across two different sheets or table \
blocks), report it as separate sections sharing the same class.
"""


def build_raw_dump(file_bytes, filename, max_chars=MAX_DUMP_CHARS):
    """Returns (text, truncated) — a plain-text rendering of everything in
    the document (all sheets, all cells, all paragraphs, all table rows),
    for Claude to read directly rather than relying on column-guessing."""
    lower = filename.lower()

    if lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        parts = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None, dtype=str)
            parts.append(f"=== Sheet: {sheet} ===")
            for _, row in df.iterrows():
                cells = ["" if pd.isna(v) else str(v).strip() for v in row.tolist()]
                if any(cells):
                    parts.append(" | ".join(cells))
        text = "\n".join(parts)

    elif lower.endswith(".csv"):
        text = file_bytes.decode("utf-8", errors="replace")

    elif lower.endswith(".pdf"):
        # Converted to plain text here, never sent to Claude as a file — a
        # PDF passed as a file gets read as page images, which costs far
        # more than the same content as plain text.
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                parts.append(f"=== Page {i} ===")
                parts.append(page.extract_text() or "")
        text = "\n".join(parts)

    elif lower.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for ti, table in enumerate(doc.tables):
            parts.append(f"=== Table {ti} ===")
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = "\n".join(parts)

    else:
        raise ValueError(f"Unsupported file type: {filename}")

    truncated = len(text) > max_chars
    return (text[:max_chars] if truncated else text), truncated


def _find_marker(haystack, marker, start_pos=0):
    """Finds `marker` in haystack[start_pos:], tolerant of whitespace
    differences (collapsed runs, wrapped lines) but not wording differences
    — this is exact-text location, not fuzzy matching. Returns (start, end)
    absolute indices into haystack, or None if not found."""
    words = marker.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, haystack[start_pos:], re.IGNORECASE)
    if not m:
        return None
    return start_pos + m.start(), start_pos + m.end()


def _pick_term_cell(line):
    """A table row flattened by build_raw_dump looks like "1 | 日本指定商品 |
    Chemicals for use in industry". Rather than keeping the whole joined
    line (which drags in row numbers and any translation column — the exact
    bug this fixes), split on "|" and keep whichever cell looks most like
    real English wording: highest ASCII ratio among cells that aren't pure
    row numbers, tie-broken by length. This is what makes the Claude
    fallback self-adapting to a table-shaped layout nobody has coded a named
    rule for yet, instead of only working for the specific layouts already
    recognized by extraction_rules.py.

    A plain paragraph line (no "|") passes through unchanged — this only
    kicks in for genuinely multi-column rows."""
    if "|" not in line:
        return line.strip()
    cells = [c.strip() for c in line.split("|") if c.strip()]
    candidates = [c for c in cells if not re.fullmatch(r"\d+\.?", c)]
    if not candidates:
        return line.strip()
    return max(candidates, key=lambda c: (extraction_rules.ascii_ratio(c), len(c)))


def _split_span_into_items(span_text):
    """A located span may be a single semicolon-joined line, or several
    table rows (each on its own line). Split on newlines first, pick the
    real wording cell out of each line (see _pick_term_cell), then split on
    semicolons within that cell — never commas, which real official terms
    routinely contain internally."""
    items = []
    for line in span_text.split("\n"):
        cell = _pick_term_cell(line)
        if not cell:
            continue
        pieces = [p.strip() for p in cell.split(";")]
        items.extend(p for p in pieces if p)
    return items


def _resolve_sections(raw_text, sections):
    items = []
    for sec in sections:
        start_marker = (sec.get("start") or "").strip()
        end_marker = (sec.get("end") or "").strip()
        if not start_marker or not end_marker:
            continue
        start_span = _find_marker(raw_text, start_marker)
        if not start_span:
            continue
        end_span = _find_marker(raw_text, end_marker, start_pos=start_span[0])
        if not end_span:
            continue
        span_text = raw_text[start_span[0]:end_span[1]]
        cls = sec.get("class", "")
        for term in _split_span_into_items(span_text):
            items.append({"class": cls, "term": term})
    return items


def _call_location_extraction(client, raw_text, filename, authority, model):
    """One API call. Returns (sections, stop_reason) — sections are
    {"class", "start", "end"} location markers, not retyped item text."""
    prompt = (
        f"Filename: {filename}\n"
        f"Target authority for this check: {authority_label(authority)}\n\n"
        "Raw extracted document content follows:\n-----\n"
        f"{raw_text}\n-----"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=LOCATION_MAX_TOKENS,
        system=[{
            "type": "text",
            "text": LOCATION_SYSTEM_PROMPT.format(authority=authority_label(authority)),
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[LOCATION_TOOL],
        tool_choice={"type": "tool", "name": "report_extraction_locations"},
        messages=[{"role": "user", "content": prompt}],
    )
    sections = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == "report_extraction_locations":
            sections = block.input.get("sections", [])
    return sections, resp.stop_reason


def _extract_locations(client, raw_text, filename, authority, model, depth=0):
    """Gets location markers for raw_text and resolves them into items. If
    the response is cut off (rare — output is tiny) the text is bisected on
    line boundaries and each half retried independently, same self-healing
    pattern as the old chunked extractor."""
    try:
        sections, stop_reason = _call_location_extraction(client, raw_text, filename, authority, model)
    except Exception:
        sections, stop_reason = [], "max_tokens"

    if stop_reason == "max_tokens" and depth < MAX_SPLIT_DEPTH:
        lines = raw_text.split("\n")
        if len(lines) > 1:
            mid = len(lines) // 2
            left, right = "\n".join(lines[:mid]), "\n".join(lines[mid:])
            return (
                _extract_locations(client, left, filename, authority, model, depth + 1)
                + _extract_locations(client, right, filename, authority, model, depth + 1)
            )
        return []

    return _resolve_sections(raw_text, sections)


def extract_wording_with_claude(
    file_bytes, filename, authority, api_key=None, model=DEFAULT_MODEL, progress_callback=None
):
    """Returns (items, truncated) where items is a list of {"class","term"}.

    Stage 1 (free): tries the plain-code layout detectors in
    extraction_rules.py. Most client files match one of a handful of common
    shapes and need no API call at all.

    Stage 2 (paid, minimal): only if stage 1 isn't confident, asks Claude
    (Haiku — locating text spans is mechanical, not a judgment call) for
    WHERE each class's text is, not to retype it. The surrounding code cuts
    the exact original text from those locations, which is both cheaper
    (output is a handful of short markers, not a full retyped list) and
    more faithful (the client's exact wording, not Claude's transcription
    of it).
    """
    try:
        rule_items, confident = extraction_rules.extract_via_rules(file_bytes, filename)
    except Exception:
        rule_items, confident = [], False

    if confident:
        if progress_callback:
            progress_callback(1, 1)
        return rule_items, False

    raw_text, truncated = build_raw_dump(file_bytes, filename)
    if not raw_text.strip():
        return [], truncated

    client = get_client(api_key)
    items = _extract_locations(client, raw_text, filename, authority, model)
    extraction_rules.flag_duplicates(items)
    # Unlike the free layouts (which simply fall back to Claude if
    # validate_items rejects them), there's no further fallback tier here —
    # so a term that still looks corrupted (see looks_corrupted) is flagged
    # for manual review rather than either silently shipped or silently
    # dropped. _pick_term_cell should prevent this in practice, but a team
    # relying on this tool for files I've never seen needs a visible signal
    # if it ever doesn't, not a silent wrong answer.
    for it in items:
        it["needs_review"] = extraction_rules.looks_corrupted(it.get("term", ""))
    if progress_callback:
        progress_callback(1, 1)
    return items, truncated
