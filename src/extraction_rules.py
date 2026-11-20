"""Free, zero-cost extraction for the handful of layouts most client files
actually use. Converts the file to plain text/structured rows with plain
code (no API), tries a few pattern-matched layouts, and only reports back
"confident" if the result passes sanity checks. Claude is the fallback for
whatever this can't confidently parse — not the default path.

Layouts handled, in order of specificity:
  A. "GOODS IN CLASS 5: item one; item two" — an inline labelled list,
     possibly several such lines/blocks for different classes.
  B. A table with an explicit class column and a wording column — the
     wording column is identified by being mostly ASCII text, which also
     means a parallel Chinese/translation column is skipped automatically.
  C. A flat one-item-per-line list under a single class stated once near
     the top of the document (e.g. "Class: 9" / "NICE Class 09").
  D. A per-class block marked by a standalone "<CL 9>" / "＜CL 09＞" row
     (full-width brackets are common in Japanese-origin filing documents),
     followed by a header row naming an "English" column, followed by that
     class's data rows until the next marker.
  E. A metadata/info-sheet table: a row stating the class (e.g. "Nice
     class" | "06"), followed by a header row naming the wording column
     (e.g. "Items", "Wording", "Description"), followed by that class's
     data rows until the next class-stating row or end of table.

If none of these confidently apply, extract_via_rules returns no items and
the caller should fall back to Claude.
"""
import io
import re

import pandas as pd
from docx import Document

from .reference_match import normalize_class

MAX_ITEM_CHARS = 200
MIN_CLASS, MAX_CLASS = 1, 45

_CLASS_LINE_RE = re.compile(
    r"(?:nice\s+)?class(?:es)?\.?\s*(?:no\.?)?\s*(\d{1,2})\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
_STANDALONE_CLASS_RE = re.compile(
    r"^\s*(?:nice\s+)?class(?:es)?\.?\s*(?:no\.?)?\s*[:\-]?\s*(\d{1,2})\s*$",
    re.IGNORECASE,
)
_BRACKETED_MARKER_RE = re.compile(r"[<＜]\s*CL\s*0?(\d{1,2})\s*[>＞]", re.IGNORECASE)


def _split_semicolons(text):
    """Splits one cell/line on ';' into sub-items. If the same piece of text
    appears more than once WITHIN this one split (e.g. a client accidentally
    typed "Soaps; Soaps"), only one copy is kept — that's a single item
    restated, not two distinct line entries. This is deliberately narrower
    than cross-row duplicate detection (flag_duplicates below), which never
    drops anything: a duplicate that spans two separate rows in the source
    file is a fact about the file worth surfacing to the reviewer, not
    something to silently resolve during a single cell's split.

    Also strips a stray leading/trailing "|" — flat_lines joins a table
    row's cells with " | ", so when a class label and its goods text sit in
    adjacent cells (e.g. "GOODS IN CLASS 5:" | "Capsules for medicines; ..."),
    the first split piece otherwise comes out as "| Capsules for medicines".
    No real official wording starts or ends with a bare pipe character, so
    this is safe to strip unconditionally."""
    parts = [p.strip().strip("|").strip() for p in (text or "").split(";")]
    parts = [p for p in parts if p]
    seen = set()
    deduped = []
    for p in parts:
        key = p.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def ascii_ratio(text):
    """Fraction of characters that are ASCII — used to pick the English
    column out of a row that also has a Chinese/translation column, without
    needing a header keyword to name which one is which. Public: also used
    by claude_extractor.py's Claude-fallback path to make it table-aware for
    layouts that have no dedicated rule yet (see _pick_term_cell there)."""
    text = text or ""
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


# --------------------------------------------------------------------------
# Step 1: file -> structured content (free, plain code)
# --------------------------------------------------------------------------

def parse_structured(file_bytes, filename):
    """Returns {"flat_lines": [str, ...], "tables": [[[cell, ...], ...], ...]}.

    flat_lines is every paragraph/row/page-line in the document, flattened
    to one string each — used by layouts A and C. tables is every
    paragraph-free 2D grid of cells found (docx tables, spreadsheet sheets,
    PDF-detected tables) — used by layout B, which needs real column
    boundaries rather than a flattened line.
    """
    lower = filename.lower()
    flat_lines = []
    tables = []

    if lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None, dtype=str)
            grid = [
                ["" if pd.isna(v) else str(v).strip() for v in row.tolist()]
                for _, row in df.iterrows()
            ]
            grid = [row for row in grid if any(row)]
            if grid:
                tables.append(grid)
                flat_lines.extend(" | ".join(row) for row in grid)

    elif lower.endswith(".csv"):
        text = file_bytes.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), header=None, dtype=str)
        grid = [
            ["" if pd.isna(v) else str(v).strip() for v in row.tolist()]
            for _, row in df.iterrows()
        ]
        grid = [row for row in grid if any(row)]
        if grid:
            tables.append(grid)
            flat_lines.extend(" | ".join(row) for row in grid)

    elif lower.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                flat_lines.extend(l for l in page_text.split("\n") if l.strip())
                for table in page.extract_tables() or []:
                    grid = [["" if c is None else str(c).strip() for c in row] for row in table]
                    grid = [row for row in grid if any(row)]
                    if grid:
                        tables.append(grid)

    elif lower.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        flat_lines.extend(p.text.strip() for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            grid = [[c.text.strip() for c in row.cells] for row in table.rows]
            grid = [row for row in grid if any(row)]
            if grid:
                tables.append(grid)
                flat_lines.extend(" | ".join(row) for row in grid)

    else:
        raise ValueError(f"Unsupported file type: {filename}")

    return {"flat_lines": flat_lines, "tables": tables}


# --------------------------------------------------------------------------
# Step 2: layout detectors (free, plain code)
# --------------------------------------------------------------------------

def _layout_a_labelled_class_lines(flat_lines):
    """"GOODS IN CLASS 5: item one; item two" and similar — may repeat for
    several classes across the document."""
    items = []
    for line in flat_lines:
        m = _CLASS_LINE_RE.search(line)
        if not m:
            continue
        cls, goods_text = m.group(1), m.group(2)
        for term in _split_semicolons(goods_text):
            items.append({"class": cls, "term": term})
    return items


def _layout_b_class_column_table(tables):
    """A table with a header row containing a "class" column and a column
    of mostly-ASCII (English) wording — picks the English column out from
    any parallel translation column automatically."""
    items = []
    for grid in tables:
        if len(grid) < 2:
            continue
        header = grid[0]
        class_col = next(
            (i for i, cell in enumerate(header) if re.search(r"class", cell, re.IGNORECASE)),
            None,
        )
        if class_col is None:
            continue

        candidate_cols = [i for i in range(len(header)) if i != class_col]
        if not candidate_cols:
            continue

        def ascii_score(col_idx):
            samples = [row[col_idx] for row in grid[1:] if col_idx < len(row) and row[col_idx]]
            if not samples:
                return -1.0
            # A pure row-number column ("No." with values "1", "2", "3"...)
            # is 100% ASCII too, so it can tie with — or beat — the real
            # wording column on ascii_ratio alone. Disqualify it outright
            # rather than letting digits compete with real text.
            if all(re.fullmatch(r"\d+\.?", s.strip()) for s in samples):
                return -1.0
            return sum(ascii_ratio(s) for s in samples) / len(samples)

        term_col = max(candidate_cols, key=ascii_score)
        if ascii_score(term_col) < 0.8:
            continue  # no column looks like plain English text — don't guess

        last_class = ""
        for row in grid[1:]:
            cls_cell = row[class_col].strip() if class_col < len(row) else ""
            if cls_cell:
                last_class = cls_cell
            term_cell = row[term_col].strip() if term_col < len(row) else ""
            if not term_cell:
                continue
            for term in (_split_semicolons(term_cell) or [term_cell]):
                items.append({"class": last_class, "term": term})
    return items


def _layout_c_single_class_list(flat_lines):
    """A standalone "Class: 9" / "NICE Class 09" line near the top, followed
    by a flat one-item-per-line list with no further structure."""
    standalone_class = None
    header_idx = None
    for i, line in enumerate(flat_lines[:10]):
        m = _STANDALONE_CLASS_RE.match(line)
        if m:
            standalone_class, header_idx = m.group(1), i
            break
    if standalone_class is None:
        return []

    items = []
    for line in flat_lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        for term in (_split_semicolons(line) or [line]):
            items.append({"class": standalone_class, "term": term})
    return items


def _layout_d_class_marker_blocks(tables):
    """A per-class block marked by a standalone "<CL 9>" / "＜CL 09＞" row
    (both bracket styles, since the same document can mix them), followed
    by a header row naming an "English" column, followed by that class's
    data rows until the next marker. Common in Japanese-origin filing
    templates with a Japanese wording column alongside the English one."""
    items = []
    for grid in tables:
        marker_rows = []
        for i, row in enumerate(grid):
            for cell in row:
                m = _BRACKETED_MARKER_RE.search(cell)
                if m:
                    marker_rows.append((i, m.group(1)))
                    break
        if not marker_rows:
            continue

        for block_idx, (marker_i, cls) in enumerate(marker_rows):
            block_end = marker_rows[block_idx + 1][0] if block_idx + 1 < len(marker_rows) else len(grid)
            header_i = marker_i + 1
            if header_i >= block_end:
                continue
            header_row = grid[header_i]
            term_col = next(
                (ci for ci, cell in enumerate(header_row) if re.search(r"english", cell, re.IGNORECASE)),
                None,
            )
            if term_col is None:
                continue  # can't confidently tell which column is the wording — don't guess
            for row in grid[header_i + 1:block_end]:
                if term_col >= len(row):
                    continue
                cell = row[term_col].strip()
                if not cell:
                    continue
                for term in (_split_semicolons(cell) or [cell]):
                    items.append({"class": cls, "term": term})
    return items


# Tried in priority order. "english"/"\ben\b" come first because a header
# naming both an English and a translation column (e.g. "Goods in CN" /
# "Goods in EN") would otherwise match the generic "goods" keyword on the
# WRONG (translation) column — only one of the two headers actually says
# which language it is, so checking for that first disambiguates them.
_HEADER_KEYWORDS = ("english", r"\ben\b", "item", "wording", "description", "goods", "service")


def _find_class_marker_in_row(row):
    """Returns the class number if this row declares a single class, either
    as one combined cell ("CLASS 14", matched via _STANDALONE_CLASS_RE) or
    as a separate label+number cell pair (["Nice class", "06", ...]).
    Returns None otherwise."""
    for cell in row:
        m = _STANDALONE_CLASS_RE.match(cell.strip())
        if m:
            return m.group(1)
    if any(re.search(r"\bclass\b", cell, re.IGNORECASE) for cell in row):
        for cell in row:
            cell = cell.strip()
            if re.fullmatch(r"\d{1,2}", cell):
                return cell
    return None


MAX_HEADER_CELL_CHARS = 20


def _find_header_term_col(row):
    """Finds the wording column by header keyword, trying the most specific
    keyword first before falling back to vaguer ones ("goods"/"service",
    which can also match a section title cell like "Goods or Services" that
    spans the row rather than naming one column). Only short, label-like
    cells count — a metadata field like "applicant's name in English" also
    contains the word "English" but is a 28-character descriptive label, not
    a 2-3 word column header; without this guard that row gets mistaken for
    the real header several rows later."""
    for kw in _HEADER_KEYWORDS:
        for ci, cell in enumerate(row):
            if len(cell.strip()) <= MAX_HEADER_CELL_CHARS and re.search(kw, cell, re.IGNORECASE):
                return ci
    return None


def _layout_e_single_class_table(tables):
    """A metadata/info-sheet table: a row stating the class (either combined
    like "CLASS 14" or split like "Nice class" | "06"), plus a header row
    naming the wording column — the header can appear once for the whole
    table (shared across all class blocks) or right after each marker, so
    it's searched for across the whole table rather than scoped to one
    block. Data rows run from each marker to the next marker (or end of
    table)."""
    items = []
    for grid in tables:
        marker_rows = [(i, cls) for i, row in enumerate(grid) if (cls := _find_class_marker_in_row(row))]
        if not marker_rows:
            continue

        header_i, term_col = None, None
        for ri, row in enumerate(grid):
            col = _find_header_term_col(row)
            if col is not None:
                header_i, term_col = ri, col
                break
        if header_i is None:
            continue  # no recognizable wording-column header anywhere — don't guess

        for block_idx, (marker_i, cls) in enumerate(marker_rows):
            block_end = marker_rows[block_idx + 1][0] if block_idx + 1 < len(marker_rows) else len(grid)
            start_i = max(marker_i + 1, header_i + 1)
            for ri in range(start_i, block_end):
                row = grid[ri]
                if term_col >= len(row):
                    continue
                cell = row[term_col].strip()
                if not cell:
                    continue
                for term in (_split_semicolons(cell) or [cell]):
                    items.append({"class": cls, "term": term})
    return items


# --------------------------------------------------------------------------
# Step 3: validation (free) — only a confidently-correct result skips Claude
# --------------------------------------------------------------------------

def looks_corrupted(term):
    """A literal "|" surviving into a term is always a sign of an un-split
    table-row-join artifact (flat_lines joins cells with " | ") — no real
    official wording ever contains one. This is a layout-independent safety
    net: it catches a bad split regardless of which rule produced it, or
    whether it came from a named layout at all vs. the Claude fallback."""
    return "|" in (term or "")


def validate_items(items):
    if not items:
        return False
    for it in items:
        norm_cls = normalize_class(it.get("class", ""))
        if not norm_cls or not norm_cls.isdigit() or not (MIN_CLASS <= int(norm_cls) <= MAX_CLASS):
            return False
        term = it.get("term", "")
        if len(term) > MAX_ITEM_CHARS or looks_corrupted(term):
            return False
    return True


def flag_duplicates(items):
    """Tags each item with is_duplicate — True if the same normalized
    wording (case/whitespace-insensitive, any class) appears more than once
    across the whole list. Never removes anything: a duplicate spanning two
    separate rows in the client's own file is a fact about that file worth
    surfacing to the reviewer, not something to resolve silently. Mutates
    and returns `items`. Matches the same term-only (class-agnostic)
    duplicate definition already used downstream in claude_matcher.py /
    streamlit_app.py, so a row flagged here stays flagged all the way
    through the pipeline."""
    counts = {}
    for it in items:
        key = (it.get("term") or "").strip().lower()
        counts[key] = counts.get(key, 0) + 1
    for it in items:
        key = (it.get("term") or "").strip().lower()
        it["is_duplicate"] = counts[key] > 1
    return items


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def extract_via_rules(file_bytes, filename):
    """Returns (items, confident). confident=False means none of the known
    layouts produced a validated result — the caller should fall back to
    Claude for this document rather than trust a low-confidence guess.
    Items that ARE returned are tagged with is_duplicate (see
    flag_duplicates) — never silently de-duplicated across rows."""
    structured = parse_structured(file_bytes, filename)

    for layout_fn, args in (
        (_layout_a_labelled_class_lines, (structured["flat_lines"],)),
        (_layout_b_class_column_table, (structured["tables"],)),
        (_layout_c_single_class_list, (structured["flat_lines"],)),
        (_layout_d_class_marker_blocks, (structured["tables"],)),
        (_layout_e_single_class_table, (structured["tables"],)),
    ):
        items = layout_fn(*args)
        if validate_items(items):
            return flag_duplicates(items), True

    return [], False
