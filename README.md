# Trademark Wording Checker

Checks a client's requested trademark goods/services wording for IPOS (Singapore) or
MyIPO (Malaysia) filings and suggests the correct standard wording where it doesn't match.

Two steps:

1. **Reading the client's document** — client files arrive in all kinds of
   formats (plain lists, numbered tables, bilingual spreadsheets, multi-country
   workbooks, instruction letters with the goods list embedded partway through,
   metadata info-sheets, per-class marker blocks). Most match one of a handful
   of common layouts and are read with plain code for free (`src/extraction_rules.py`).
   Claude is only asked to step in — and even then, only to locate the relevant
   text, not retype it — when none of those layouts confidently apply.
2. **Checking each item** — whether an item's wording exactly matches a real
   official term is a yes/no fact, so it's checked for free against the
   downloaded IPOS/MyIPO wording lists (`data/reference/`) — no API call
   needed for an exact match, or a match that exists under a different class
   (flagged as "standard wording, wrong class" and auto-corrected). Only
   items with no match anywhere in the official list go to Claude, which
   suggests a correction from its own knowledge of IPOS/MyIPO/NICE
   conventions, framed as an experienced trademark prosecution agent — since
   there's no official "closest correction" to look up for those, treat that
   part as a strong first pass, not a verified source.

## Setup

```
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste in your Anthropic API key
```

## Run

```
streamlit run streamlit_app.py
```

## Usage

Upload a client's goods/services file (Word, PDF, or Excel), review/edit the
extracted items, run the check, and download an Excel report — non-standard
rows in red, exact duplicates highlighted, plus a "Class List" sheet grouping
the final wording by NICE class for pasting straight into a filing form.

## Project structure

- `streamlit_app.py` — the app
- `src/extraction_rules.py` — free, zero-cost document extraction for common layouts (labelled lists, class-column tables, per-class marker blocks, metadata info-sheets)
- `src/claude_extractor.py` — falls back to Claude only when `extraction_rules.py` isn't confident; asks for text locations, not a retyped list
- `src/reference_match.py` — free local matching against the official IPOS/MyIPO wording lists
- `src/claude_matcher.py` — pre-filters through `reference_match.py`, sends only non-standard items to Claude
- `src/claude_client.py` — shared Anthropic client construction
- `data/reference/` — the downloaded official IPOS/MyIPO wording lists (~117k / ~136k terms)
- `scripts/test_local_extraction.py`, `scripts/test_local_matching.py` — zero-API test harnesses for the free extraction and matching steps, useful for checking a real client file's layout/accuracy before spending anything on Claude
