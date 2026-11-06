# Trademark Wording Checker

Checks a client's requested trademark goods/services wording against an official
classification list (IPOS Singapore or MyIPO Malaysia) and suggests the correct
standard wording where it doesn't match.

Claude does the actual work in two steps:

1. **Reading the client's document** — client files arrive in all kinds of
   formats (plain lists, numbered tables, bilingual spreadsheets, multi-country
   workbooks, instruction letters with the goods list embedded partway through).
   Claude reads the raw document content and extracts the wording items and
   their NICE class itself, rather than relying on column-guessing.
2. **Reviewing the wording** — a local TF-IDF index shortlists the closest
   official terms for each item (the reference lists run to 100k+ rows, too
   large to hand to Claude directly), and Claude reviews that shortlist to
   decide whether the wording is standard and, if not, what the correct
   official wording is.

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

- **Upload official list** — load or refresh the IPOS/MyIPO reference list
  whenever the authority publishes an update. Saved as a versioned snapshot
  under `data/reference/` so you don't need to re-upload it every session.
- **Check client wording** — upload a client's goods/services file (Word,
  PDF, or Excel) and get a per-item standard/non-standard verdict with
  suggested wording, downloadable as an Excel report.

## Project structure

- `streamlit_app.py` — the app (two pages: check wording, upload official list)
- `src/claude_extractor.py` — Claude-based document reading/extraction
- `src/claude_matcher.py` — Claude-based wording review
- `src/matcher.py` — local TF-IDF/fuzzy shortlisting used to narrow candidates before Claude review
- `src/parsers.py` — fallback heuristic parsing for the official-list upload flow
- `src/reference_store.py` — versioned storage for the official reference lists
