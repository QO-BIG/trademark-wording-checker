"""File parsers that turn an uploaded reference list or client wording file
into a plain list of rows: {"class": "9", "term": "mobile phones"} etc.

Everything here is heuristic-first, confirm-by-human-second: parsing never
commits data on its own, it only proposes rows for a user to review on the
upload preview screen.
"""
import io
import re

import pandas as pd
from docx import Document
from pypdf import PdfReader

CLASS_HEADER_RE = re.compile(r"^\s*class\s*(\d{1,2})\b", re.IGNORECASE)

# Matches official-list PDF exports shaped like "12 1 unprocessed resins..."
# i.e. a running row number, then a NICE class number, then the term text.
NUMBERED_TABLE_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d{1,2})\s+(.+?)\s*$")

# Some pages of the same export split the running row number onto its own
# line(s) instead (e.g. "10720" / "2" / "37 installation and maintenance of
# hardware..."), so the content line only starts with the class number.
CLASS_ONLY_ROW_RE = re.compile(r"^\s*(\d{1,2})\s+(.+?)\s*$")

# A row-number fragment on its own line, from the split-numbering pages above.
STANDALONE_DIGITS_RE = re.compile(r"^\s*\d+\s*$")

MAX_NICE_CLASS = 45

PDF_BOILERPLATE_RE = re.compile(
    r"^(list of trademark classes|class:\s*all|goods and services:\s*all|"
    r"no\.?\s*class\s*goods and services|page \d+ of \d+)\s*$",
    re.IGNORECASE,
)

# Header names we'll guess a tabular column maps to.
CLASS_COL_HINTS = ["class", "nice class", "class no", "class number"]
TERM_COL_HINTS = [
    "term", "wording", "description", "goods", "services",
    "goods and services", "goods/services", "item", "specification",
    "goods items",
]
HEADER_ROW_SCAN_LIMIT = 15


def _best_column(columns, hints, prefer=(), avoid=()):
    """`prefer`/`avoid` substrings break ties on bilingual sheets, e.g. a
    client's wording list with both 'Designated Items in Chinese' and
    'Designated Items in English' columns — we want the English one.
    """
    lowered = {c: str(c).strip().lower() for c in columns}

    def matches(hint):
        return [col for col, low in lowered.items() if hint in low]

    for hint in hints:
        candidates = [col for col, low in lowered.items() if low == hint]
        if candidates:
            preferred = [c for c in candidates if any(p in lowered[c] for p in prefer)]
            return (preferred or candidates)[0]

    for hint in hints:
        candidates = matches(hint)
        if not candidates:
            continue
        preferred = [c for c in candidates if any(p in lowered[c] for p in prefer)]
        if preferred:
            return preferred[0]
        not_avoided = [c for c in candidates if not any(a in lowered[c] for a in avoid)]
        if not_avoided:
            return not_avoided[0]
        return candidates[0]
    return None


def _score_header_row(values):
    # Only consider short, column-name-like cells — a long disclaimer/title
    # paragraph can coincidentally contain words like "class" or "goods"
    # too, and must not be mistaken for the real header row.
    cells = [
        str(v).strip().lower() for v in values
        if str(v).strip().lower() not in ("", "nan") and len(str(v).strip()) <= 40
    ]
    class_hit = any(any(hint in cell for hint in CLASS_COL_HINTS) for cell in cells)
    term_hit = any(any(hint in cell for hint in TERM_COL_HINTS) for cell in cells)
    return (1 if class_hit else 0) + (1 if term_hit else 0)


def _find_header_row(raw_df):
    """Official lists often have disclaimer/title rows before the real
    header (see IPOS classification.xlsx). Scan the first few rows and pick
    whichever looks most like a header instead of assuming row 0.
    """
    best_idx, best_score = 0, 0
    for idx in range(min(HEADER_ROW_SCAN_LIMIT, len(raw_df))):
        score = _score_header_row(raw_df.iloc[idx].tolist())
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx if best_score > 0 else 0


def parse_tabular(file_bytes, filename):
    """Parse .xlsx/.xls/.csv into a DataFrame and guess class/term columns."""
    if filename.lower().endswith(".csv"):
        raw_df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    else:
        raw_df = pd.read_excel(io.BytesIO(file_bytes), header=None)

    header_idx = _find_header_row(raw_df)
    columns = [str(c).strip() for c in raw_df.iloc[header_idx].tolist()]
    df = raw_df.iloc[header_idx + 1:].copy()
    df.columns = columns
    df = df.dropna(how="all")

    guessed_class_col = _best_column(columns, CLASS_COL_HINTS)
    guessed_term_col = _best_column(
        columns, TERM_COL_HINTS, prefer=("english",), avoid=("chinese", "malay", "translation")
    )

    if guessed_class_col:
        # Merged "Class" cells (one value per class block) export as blanks
        # for every row after the first — carry the class number down.
        df[guessed_class_col] = df[guessed_class_col].ffill().infer_objects(copy=False)

    df = df.fillna("").infer_objects(copy=False)
    return {
        "kind": "tabular",
        "columns": columns,
        "preview_rows": df.head(20).to_dict(orient="records"),
        "all_rows": df.to_dict(orient="records"),
        "guessed_class_col": guessed_class_col,
        "guessed_term_col": guessed_term_col,
    }


def rows_from_tabular(all_rows, class_col, term_col):
    rows = []
    for r in all_rows:
        term = str(r.get(term_col, "")).strip()
        if not term or term.lower() == "nan":
            continue
        cls = str(r.get(class_col, "")).strip() if class_col else ""
        if cls.lower() == "nan":
            cls = ""
        if cls.endswith(".0"):
            cls = cls[:-2]
        rows.append({"class": cls, "term": term})
    return rows


def _lines_to_class_term_rows(lines):
    """Fallback heuristic for PDF/DOCX text with no recognizable table: a
    'Class N' line starts a new class, subsequent non-empty lines until the
    next class header are terms. If no class headers are found at all,
    every non-empty line is a term with an empty class, and the user fixes
    it up in the preview screen.
    """
    rows = []
    current_class = ""
    seen_any_class_header = False
    for raw in lines:
        line = raw.strip().strip("•-–\t ")
        if not line:
            continue
        m = CLASS_HEADER_RE.match(line)
        if m:
            current_class = m.group(1)
            seen_any_class_header = True
            continue
        rows.append({"class": current_class, "term": line})
    return rows, seen_any_class_header


MIN_NUMBERED_ROWS_TO_CONFIRM_FORMAT = 20
# A genuine wrapped line is at most 1-2 extra lines. If a stretch of the
# document stops matching the numbered-row pattern for longer than this
# (e.g. a layout change partway through a multi-thousand-page PDF), it's no
# longer a line wrap — it's a different section — so stop gluing text onto
# the previous row instead of silently merging unrelated content into one
# unusable blob entry.
MAX_CONTINUATION_LINES = 3


def _rows_from_numbered_table(lines):
    """Official-list PDF exports (e.g. 'NO. CLASS GOODS AND SERVICES' with
    rows like '12 1 unprocessed resins...') repeated across thousands of
    pages. Detect and parse that shape directly — much more reliable than
    the generic heuristic, and the common case for large reference PDFs.

    Long terms often wrap onto a second line with no leading numbers (the
    PDF just breaks the sentence) — any non-matching, non-boilerplate line
    that follows a matched row is treated as a continuation of that row's
    term, bounded by MAX_CONTINUATION_LINES so a genuine format change
    elsewhere in the document can't cascade into one giant merged entry.
    """
    rows = []
    matched_count = 0
    continuation_streak = 0
    for raw in lines:
        line = raw.strip()
        if not line or PDF_BOILERPLATE_RE.match(line) or STANDALONE_DIGITS_RE.match(line):
            continue

        m = NUMBERED_TABLE_ROW_RE.match(line)
        if m and 1 <= int(m.group(2)) <= MAX_NICE_CLASS:
            rows.append({"class": m.group(2), "term": m.group(3).strip()})
            matched_count += 1
            continuation_streak = 0
            continue

        m2 = CLASS_ONLY_ROW_RE.match(line)
        if m2 and 1 <= int(m2.group(1)) <= MAX_NICE_CLASS:
            rows.append({"class": m2.group(1), "term": m2.group(2).strip()})
            matched_count += 1
            continuation_streak = 0
            continue

        if rows and continuation_streak < MAX_CONTINUATION_LINES:
            rows[-1]["term"] = f"{rows[-1]['term']} {line}".strip()
            continuation_streak += 1
        # else: too many consecutive non-matching lines — likely a
        # different section of the document, not a wrapped line. Drop it
        # rather than corrupting the previous row.

    if matched_count < MIN_NUMBERED_ROWS_TO_CONFIRM_FORMAT:
        return None  # doesn't look like this format
    return rows


def parse_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(text.split("\n"))

    table_rows = _rows_from_numbered_table(lines)
    if table_rows is not None:
        return {
            "kind": "text",
            "all_rows": table_rows,
            "preview_rows": table_rows[:20],
            "had_class_headers": True,
        }

    rows, had_class_headers = _lines_to_class_term_rows(lines)
    return {
        "kind": "text",
        "all_rows": rows,
        "preview_rows": rows[:20],
        "had_class_headers": had_class_headers,
    }


def parse_docx(file_bytes, filename):
    doc = Document(io.BytesIO(file_bytes))
    lines = [p.text for p in doc.paragraphs]

    table_rows = _rows_from_numbered_table(lines)
    if table_rows is not None:
        return {
            "kind": "text",
            "all_rows": table_rows,
            "preview_rows": table_rows[:20],
            "had_class_headers": True,
        }

    rows, had_class_headers = _lines_to_class_term_rows(lines)
    return {
        "kind": "text",
        "all_rows": rows,
        "preview_rows": rows[:20],
        "had_class_headers": had_class_headers,
    }


def format_rows_as_text(rows, include_class=True):
    lines = []
    for r in rows:
        term = r.get("term", "")
        if include_class:
            cls = r.get("class", "") or ""
            lines.append(f"{cls} | {term}" if cls else f" | {term}")
        else:
            lines.append(term)
    return "\n".join(lines)


def parse_text_as_rows(text, include_class=True):
    rows = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if include_class and "|" in line:
            cls, term = line.split("|", 1)
            cls, term = cls.strip(), term.strip()
        else:
            cls, term = "", line.strip()
        if term:
            rows.append({"class": cls, "term": term})
    return rows


def parse_upload(file_bytes, filename):
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls", ".csv")):
        return parse_tabular(file_bytes, filename)
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    if lower.endswith(".docx"):
        return parse_docx(file_bytes, filename)
    raise ValueError(f"Unsupported file type: {filename}")
