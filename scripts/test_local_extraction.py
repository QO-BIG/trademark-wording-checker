"""Stand-alone, zero-API test of the free rule-based extraction step only.

Lets you check whether your real client files match one of the recognized
free layouts, WITHOUT calling Claude at all — no API key needed, no cost.
If a file doesn't qualify as a confident free extraction, this also shows
you WHY (which layout almost matched, and what failed validation), so you
can tell whether the gap is worth fixing (e.g. a header that says "Cls."
instead of "Class") or is a genuinely unusual layout that should fall back
to Claude in the real app.

Usage:
    python scripts/test_local_extraction.py <file.xlsx|.docx|.csv|.pdf>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import extraction_rules as er


def _print_items(items, label):
    print(f"\n  {label}: {len(items)} item(s)")
    for it in items[:15]:
        print(f"    class={it['class']!r:8} term={it['term']!r}")
    if len(items) > 15:
        print(f"    ... and {len(items) - 15} more")


def _why_invalid(items):
    if not items:
        return "no items found"
    for it in items:
        norm_cls = er.normalize_class(it.get("class", ""))
        if not norm_cls or not norm_cls.isdigit() or not (er.MIN_CLASS <= int(norm_cls) <= er.MAX_CLASS):
            return f"item has an invalid/out-of-range class: class={it['class']!r} term={it['term']!r}"
        if len(it.get("term", "")) > er.MAX_ITEM_CHARS:
            return f"an item is over {er.MAX_ITEM_CHARS} chars (likely a missed split): {it['term'][:80]!r}..."
    return "unknown"  # shouldn't happen if validate_items agrees


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    print(f"Parsing {filename} into structured content (free, no API)...")
    structured = er.parse_structured(file_bytes, filename)
    print(f"  {len(structured['flat_lines'])} flattened lines, {len(structured['tables'])} table(s) found")

    print("\n" + "=" * 72)
    print("Trying each free layout in order:")
    print("=" * 72)

    layout_a = er._layout_a_labelled_class_lines(structured["flat_lines"])
    valid_a = er.validate_items(layout_a)
    _print_items(layout_a, "Layout A — 'GOODS IN CLASS N: ...' inline list")
    print(f"    -> {'VALID (would be used)' if valid_a else 'not valid: ' + _why_invalid(layout_a)}")

    layout_b = er._layout_b_class_column_table(structured["tables"])
    valid_b = er.validate_items(layout_b)
    _print_items(layout_b, "Layout B — table with a class column")
    print(f"    -> {'VALID (would be used)' if valid_b else 'not valid: ' + _why_invalid(layout_b)}")

    layout_c = er._layout_c_single_class_list(structured["flat_lines"])
    valid_c = er.validate_items(layout_c)
    _print_items(layout_c, "Layout C — standalone 'Class: N' + flat list")
    print(f"    -> {'VALID (would be used)' if valid_c else 'not valid: ' + _why_invalid(layout_c)}")

    layout_d = er._layout_d_class_marker_blocks(structured["tables"])
    valid_d = er.validate_items(layout_d)
    _print_items(layout_d, "Layout D — '<CL N>' marker blocks + English-header column")
    print(f"    -> {'VALID (would be used)' if valid_d else 'not valid: ' + _why_invalid(layout_d)}")

    layout_e = er._layout_e_single_class_table(structured["tables"])
    valid_e = er.validate_items(layout_e)
    _print_items(layout_e, "Layout E — 'Nice class: N' metadata table + header row")
    print(f"    -> {'VALID (would be used)' if valid_e else 'not valid: ' + _why_invalid(layout_e)}")

    print("\n" + "=" * 72)
    items, confident = er.extract_via_rules(file_bytes, filename)
    if confident:
        dup_count = sum(1 for it in items if it.get("is_duplicate"))
        print(f"RESULT: Confident free extraction — {len(items)} item(s). Zero Claude calls needed.")
        if dup_count:
            print(f"  {dup_count} item(s) flagged as duplicate wording (kept, NOT removed) — see below:")
            for it in items:
                if it.get("is_duplicate"):
                    print(f"    DUPLICATE  class={it['class']!r:8} term={it['term']!r}")
    else:
        print(
            "RESULT: No layout confidently matched. In the real app, this file would fall "
            "back to Claude (a small, cheap location-finding call) — not an error, just not free."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
