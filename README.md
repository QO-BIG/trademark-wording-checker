# Trademark Wording Checker

Checks a client's requested trademark goods/services wording for IPOS (Singapore) or
MyIPO (Malaysia) filings and suggests the correct standard wording where it doesn't match.

Claude does the actual work in two steps, the same as pasting the document into a
Claude chat and asking it to check it — there's no downloaded official list behind
this, so treat suggestions as a strong first pass, not a verified source:

1. **Reading the client's document** — client files arrive in all kinds of
   formats (plain lists, numbered tables, bilingual spreadsheets, multi-country
   workbooks, instruction letters with the goods list embedded partway through).
   Claude reads the raw document content and extracts the wording items and
   their NICE class itself, rather than relying on column-guessing.
2. **Reviewing the wording** — Claude judges each item's standard/non-standard
   status and suggests corrected wording from its own knowledge of IPOS/MyIPO
   classification conventions, framed as an experienced trademark prosecution
   agent rather than a generic matcher.

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
- `src/claude_extractor.py` — Claude-based document reading/extraction (chunked for large documents)
- `src/claude_matcher.py` — Claude-based wording review
- `src/claude_client.py` — shared Anthropic client construction
