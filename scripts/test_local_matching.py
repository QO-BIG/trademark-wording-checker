"""Stand-alone, zero-API test of the free local matching step only.

Lets you check accuracy of the standard/non-standard/wrong-class calls
against the real IPOS/MyIPO reference lists, using your own real wording
list, WITHOUT calling Claude at all — no extraction, no review, no API key
needed, no cost. Stops exactly where the real app would hand off to Claude:
it tells you which items are free matches (and what they resolved to) and
which items WOULD be sent to Claude for a suggestion, without sending them.

Usage:
    python scripts/test_local_matching.py <file.xlsx|file.csv> <ipos|myipo>

Input file needs a "class" column and a "term" column (case-insensitive
header match). This is the same two-column shape the app's editable table
uses after extraction — so you can type one up by hand in Excel, or export
one from a prior "check_rows" session if you have one.

Output: a console summary + a full per-item breakdown written to
<file>-local-match-report.csv next to the input file.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import reference_match as rm


def load_items(path):
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    elif lower.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError("Input file must be .xlsx, .xls, or .csv")

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "class" not in df.columns or "term" not in df.columns:
        raise ValueError(
            f"Expected columns 'class' and 'term' (case-insensitive), found: {list(df.columns)}"
        )

    df = df.fillna("")
    return [
        {"class": str(row["class"]).strip(), "term": str(row["term"]).strip()}
        for _, row in df.iterrows()
        if str(row["term"]).strip()
    ]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    file_path, authority = sys.argv[1], sys.argv[2].strip().lower()
    if authority not in ("ipos", "myipo"):
        print(f"Authority must be 'ipos' or 'myipo', got: {authority!r}")
        sys.exit(1)

    items = load_items(file_path)
    print(f"Loaded {len(items)} items from {file_path}")

    print(f"Loading official {authority.upper()} reference list...")
    rows = rm.load_authority_reference(authority)
    index = rm.ReferenceIndex(rows)
    print(f"Reference list loaded: {len(rows)} official terms.\n")

    results = rm.classify_items_locally(items, index)

    free_count = sum(1 for r in results if not r["needs_agent"])
    agent_count = sum(1 for r in results if r["needs_agent"])
    wrong_class_count = sum(1 for r in results if r["status"] == "standard_wrong_class")
    dup_count = sum(1 for r in results if r["is_duplicate"])

    print("=" * 72)
    print(f"{len(results)} items total (after splitting any semicolon-joined lines)")
    print(f"  {free_count} resolved for FREE ({wrong_class_count} of those were wrong-class corrections)")
    print(f"  {agent_count} would be sent to Claude for a suggestion (NOT sent — this is a local-only test)")
    print(f"  {dup_count} flagged as exact duplicates")
    print("=" * 72)

    print(f"\n{'Class':<8}{'Status':<34}{'Item':<60}")
    print("-" * 102)
    for r in results:
        status_label = {
            "standard": "Standard",
            "standard_wrong_class": f"Standard (wrong class -> {r['correct_class']})",
            "non_standard": "-> WOULD GO TO CLAUDE",
        }[r["status"]]
        term_display = r["input"] if len(r["input"]) <= 57 else r["input"][:54] + "..."
        print(f"{r['class']:<8}{status_label:<34}{term_display:<60}")

    report_path = os.path.splitext(file_path)[0] + "-local-match-report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Term", "Status", "Official Term", "Correct Class", "Duplicate", "Needs Claude"])
        for r in results:
            writer.writerow([
                r["class"], r["input"], r["status"], r["official_term"] or "",
                r["correct_class"] or "", "Yes" if r["is_duplicate"] else "",
                "Yes" if r["needs_agent"] else "",
            ])
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
