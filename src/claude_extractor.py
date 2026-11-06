"""Claude-driven extraction of a goods/services wording list from a client's
document, in whatever format they happened to send it.

Column-guessing heuristics break down fast in practice: a client's file
might be a clean one-item-per-line list, a numbered table, a bilingual
spreadsheet with English/Chinese columns, a multi-sheet workbook with one
sheet per designated country, or — seen in a real example — an instruction
letter where the goods list is buried as "GOODS IN CLASS 5: item1; item2;
..." inside a two-column metadata table, surrounded by applicant details
and boilerplate filing remarks. Rather than special-case every shape, the
raw document content (every sheet/paragraph/table cell) is dumped to text
and Claude is asked to pick out the actual wording items and their class.
"""
import io

import pandas as pd
from docx import Document
from pypdf import PdfReader

from .claude_client import get_client

DEFAULT_MODEL = "claude-sonnet-5"
MAX_DUMP_CHARS = 60000

EXTRACTION_TOOL = {
    "name": "report_extracted_wording",
    "description": "Report every goods/services wording item found in the document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {
                            "type": "string",
                            "description": "NICE classification class number for this item, as stated or clearly implied by the document (e.g. a 'GOODS IN CLASS 5:' label, or a class column/header covering a block of rows). Empty string if not determinable.",
                        },
                        "term": {
                            "type": "string",
                            "description": "The exact English goods/services wording for this single item, split out from any surrounding list punctuation, verbatim as written (do not rewrite, correct, or standardize it).",
                        },
                    },
                    "required": ["class", "term"],
                },
            }
        },
        "required": ["items"],
    },
}

EXTRACTION_SYSTEM_PROMPT = """You are extracting a goods/services wording list from a client's trademark \
filing document for a {authority} application. Clients send this in wildly different formats — a clean \
one-item-per-line list, a numbered table, an application instruction letter with the goods list embedded \
part-way through (e.g. a line like "GOODS IN CLASS 5: item one; item two; item three" inside a metadata \
table, mixed in with applicant details and boilerplate remarks), a bilingual spreadsheet with English and \
Chinese columns side by side, a multi-sheet workbook with one sheet per designated country, a spreadsheet \
with a merged "Class" column where the class number is only written once per block of rows, etc.

Read the raw extracted content below (every sheet / paragraph / table cell from the original file) and \
output ONLY the actual individual goods/services wording items, each with its NICE class if the document \
states or clearly implies one. Rules:
- Split semicolon/comma-separated lists embedded in a single cell or line (e.g. "GOODS IN CLASS 5: a; b; c") \
into one item each, all carrying that same class.
- Use the ENGLISH wording only — ignore any Chinese or other-language translation column or text.
- Ignore anything that is not a goods/services item: applicant name/address/dates, headers, column titles, \
page numbers, running row numbers, filing instructions, disclaimers, trademark names.
- If the document has multiple sheets/sections for different designated countries, extract only the one \
relevant to {authority} if you can tell which is which; otherwise extract everything you find and leave \
class empty where unclear rather than guessing.
- Preserve each item's wording exactly as written — do not rewrite, correct spelling, or standardize it. A \
separate step reviews wording quality; your only job here is faithful extraction.
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
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for i, page in enumerate(reader.pages, 1):
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


def extract_wording_with_claude(file_bytes, filename, authority, api_key=None, model=DEFAULT_MODEL):
    """Returns (items, truncated) where items is a list of {"class","term"}."""
    raw_text, truncated = build_raw_dump(file_bytes, filename)
    if not raw_text.strip():
        return [], truncated

    client = get_client(api_key)
    prompt = (
        f"Filename: {filename}\n"
        f"Target authority for this check: {authority.upper()}\n\n"
        "Raw extracted document content follows:\n-----\n"
        f"{raw_text}\n-----"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=EXTRACTION_SYSTEM_PROMPT.format(authority=authority.upper()),
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "report_extracted_wording"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "report_extracted_wording":
            return block.input.get("items", []), truncated
    return [], truncated
