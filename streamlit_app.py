"""Streamlit app: upload a client's goods/services wording and get
Claude-reviewed suggestions — Claude judges standard/non-standard and
suggests corrections from its own knowledge of IPOS/MyIPO conventions, the
same as pasting the document into a Claude chat and asking it to check it.
There is no local reference list; nothing here is grounded in a downloaded,
verifiable official list.

Run with: streamlit run streamlit_app.py
"""
import io
import os
import subprocess

import pandas as pd
import streamlit as st
from openpyxl.styles import Font, PatternFill

from src import claude_extractor, claude_matcher

st.set_page_config(page_title="Trademark Wording Checker", page_icon="⚖️", layout="wide")


def get_deployed_commit():
    """Short git commit hash of the code currently running — lets you
    confirm a Streamlit Cloud reboot actually picked up the latest push,
    instead of guessing whether an error is a new bug or a stale deploy."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_api_key():
    """Reads from .streamlit/secrets.toml first, then the environment —
    never shown or entered in the UI."""
    try:
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        secret_key = ""
    return (secret_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()


def sidebar():
    st.sidebar.title("Trademark Wording Checker")

    if get_api_key():
        st.sidebar.success("Anthropic API key configured.")
    else:
        st.sidebar.warning(
            "No Anthropic API key set. Add ANTHROPIC_API_KEY to .streamlit/secrets.toml to enable Claude."
        )

    commit = get_deployed_commit()
    st.sidebar.caption(f"Build: {commit}" if commit else "Build: unknown (not a git checkout)")


def _normalize_literal(text):
    """Lowercase/whitespace/trailing-punctuation normalization for exact
    duplicate detection — literal text match only, never similar/
    overlapping wording."""
    t = " ".join(text.strip().lower().split())
    return t.rstrip(";:.,")


def check_page():
    st.title("Check client wording")
    st.caption(
        "Upload the client's requested goods/services wording. Claude reads the document directly and "
        "reviews each item against its own knowledge of IPOS/MyIPO classification conventions — there's no "
        "downloaded official list behind this, so treat suggestions as a strong first pass, not a verified source."
    )

    authority = st.selectbox("Check against", ["ipos", "myipo"], format_func=str.upper, key="check_authority_select")

    uploaded = st.file_uploader("Client wording file", type=["xlsx", "xls", "csv", "pdf", "docx"], key="check_file")
    if uploaded is None:
        return

    api_key = get_api_key()
    if not api_key:
        st.error("No Anthropic API key configured — add ANTHROPIC_API_KEY to .streamlit/secrets.toml first.")
        return

    cache_key = (uploaded.name, uploaded.size, authority)
    if st.session_state.get("check_cache_key") != cache_key:
        progress = st.progress(0.0, text=f"Reading {uploaded.name} with Claude...")

        def on_extract_progress(done, total):
            progress.progress(done / total if total else 1.0, text=f"Reading {uploaded.name} with Claude... part {done}/{total}")

        try:
            items, truncated = claude_extractor.extract_wording_with_claude(
                uploaded.getvalue(), uploaded.name, authority, api_key=api_key,
                progress_callback=on_extract_progress,
            )
        except Exception as e:
            st.error(f"Claude document extraction failed: {e}")
            return
        progress.empty()
        if truncated:
            st.warning(
                "This document was very large and had to be truncated before sending to Claude — "
                "items near the end of the file may be missing. Consider splitting it into smaller files."
            )
        st.session_state["check_rows"] = [
            {"class": it.get("class", ""), "term": it.get("term", "")}
            for it in items if str(it.get("term", "")).strip()
        ]
        st.session_state["check_cache_key"] = cache_key
        st.session_state["check_filename"] = uploaded.name
        st.session_state.pop("check_results", None)

    filename = st.session_state["check_filename"]
    uniq = f"{filename}_{uploaded.size}"
    rows = st.session_state["check_rows"]

    st.caption("Edit any row below to fix misparsed entries before running the check.")
    edited_df = st.data_editor(
        pd.DataFrame(rows), num_rows="dynamic", width="stretch", key=f"check_editor_{uniq}"
    )

    if st.button("Run check", type="primary"):
        final_rows = [r for r in edited_df.fillna("").to_dict("records") if str(r.get("term", "")).strip()]
        if not final_rows:
            st.error("No wording found to check.")
            return

        terms = [r["term"] for r in final_rows]
        classes = [r.get("class", "") for r in final_rows]

        progress = st.progress(0.0, text="Reviewing with Claude...")

        def on_progress(done, total):
            progress.progress(done / total if total else 1.0, text=f"Reviewing with Claude... {done}/{total}")

        try:
            results = claude_matcher.match_terms_with_claude(
                terms, authority, client_classes=classes, api_key=api_key, progress_callback=on_progress
            )
        except Exception as e:
            st.error(f"Claude review failed: {e}")
            return
        progress.empty()

        for r, cls in zip(results, classes):
            r["class"] = cls

        _flag_exact_duplicates(results)
        _add_cross_class_notes(results)

        st.session_state["check_results"] = {"authority": authority, "filename": filename, "results": results}

    if "check_results" in st.session_state:
        render_results(st.session_state["check_results"])


def _flag_exact_duplicates(results):
    """Flags items whose wording is an EXACT duplicate of another item in
    this same batch — literal, normalized-text match only, never
    similar/overlapping wording — so reviewers can catch accidental
    repeats in a client's list."""
    groups = {}
    for r in results:
        key = _normalize_literal(str(r.get("input", "")))
        groups.setdefault(key, []).append(r)
    for group in groups.values():
        is_dup = len(group) > 1
        for r in group:
            r["is_duplicate"] = is_dup


def _add_cross_class_notes(results):
    """If the best-fitting official term belongs to a different NICE class
    than the client specified for this item, note it — the wording itself
    may be standard, just classified under a different class than filed."""
    for r in results:
        if not r["suggestions"]:
            continue
        client_cls = (r.get("class") or "").strip().lstrip("0")
        official_cls = (r["suggestions"][0].get("class") or "").strip().lstrip("0")
        if client_cls and official_cls and client_cls != official_cls:
            official_cls_display = r["suggestions"][0]["class"]
            if r["status"] == "standard":
                note = f"Standard wording, but classified under Class {official_cls_display}, not the stated Class {r['class']}."
            else:
                note = f"Suggested wording is classified under Class {official_cls_display}, not the stated Class {r['class']}."
            r["reasoning"] = f"{r.get('reasoning', '')} {note}".strip()


def render_results(data):
    results = data["results"]
    non_standard_count = sum(1 for r in results if r["status"] == "non_standard")

    st.subheader("Results")

    duplicate_count = sum(1 for r in results if r.get("is_duplicate"))
    st.write(
        f"{data['filename']} checked against **{data['authority'].upper()}** — "
        f"{len(results)} items, **{non_standard_count} non-standard**"
        + (f", **{duplicate_count} exact duplicate(s)**" if duplicate_count else "")
        + "."
    )
    if duplicate_count:
        st.warning(
            f"{duplicate_count} item(s) are exact duplicates of another item in this list "
            "(same wording, not just similar) — see the Duplicate column below."
        )

    display_rows = [
        {
            "Class": r.get("class", ""),
            "Client wording": r["input"],
            "Status": "Standard" if r["status"] == "standard" else "Non-standard",
            "Suggestion": r["suggestions"][0]["term"] if r["suggestions"] else "",
            "Alternatives": "; ".join(s["term"] for s in r["suggestions"][1:]),
            "Duplicate": "Yes" if r.get("is_duplicate") else "",
            "Reasoning": r.get("reasoning", ""),
        }
        for r in results
    ]

    def highlight_row(row):
        style = ""
        if row["Status"] == "Non-standard":
            style = "color: #b00020; font-weight: 600;"
        if row["Duplicate"] == "Yes":
            style += " background-color: #fff3cd;"
        return [style] * len(row)

    # A Styler's colors only render for read-only columns in st.data_editor —
    # editable cells are live input widgets and can't carry custom CSS. Status
    # and Duplicate are computed, not meant to be hand-typed over, so they're
    # the ones made read-only here in exchange for genuine red/yellow styling;
    # everything else (class, wording, suggestion, reasoning) stays editable.
    styled = pd.DataFrame(display_rows).style.apply(highlight_row, axis=1)

    st.caption(
        "Status/Duplicate are read-only (colored); edit any other field before downloading the report."
    )
    edited_df = st.data_editor(
        styled,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Status": st.column_config.Column(disabled=True),
            "Duplicate": st.column_config.Column(disabled=True),
        },
        key=f"results_editor_{data['filename']}",
    )

    st.download_button(
        "Download Excel report",
        data=build_excel_report(edited_df),
        file_name=f"wording-check-{data['authority']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


NON_STANDARD_FONT = Font(color="CC0000")
DUPLICATE_FILL = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")


def build_excel_report(edited_df):
    """Mirrors the working 'Goods Items / Your Suggestion' format: class
    shown once per block, row number restarting at 1 per class. Built from
    the (possibly user-edited) results table, so manual corrections are
    reflected in the download. Non-standard rows get red font; rows that
    are an EXACT duplicate of another row's wording (not similar/
    overlapping — literal text match only) get a yellow highlight."""
    out_rows = []
    row_flags = []  # (is_non_standard, is_duplicate), parallel to out_rows
    running_no = {}
    last_class = object()
    for _, row in edited_df.fillna("").iterrows():
        cls = str(row.get("Class", "")).strip()
        running_no[cls] = running_no.get(cls, 0) + 1
        item_text = str(row.get("Client wording", "")).strip()
        suggestion_text = str(row.get("Suggestion", "")).strip()
        if item_text.endswith(";") and suggestion_text and not suggestion_text.endswith(";"):
            suggestion_text += ";"
        out_rows.append({
            "Classes": cls if cls != last_class else "",
            "No.": running_no[cls],
            "Goods Items": item_text,
            "Your Suggestion": suggestion_text,
            "Alternatives": str(row.get("Alternatives", "")).strip(),
            "Notes": str(row.get("Reasoning", "")).strip(),
        })
        last_class = cls
        row_flags.append((
            "Non-standard" in str(row.get("Status", "")),
            "Yes" in str(row.get("Duplicate", "")),
        ))

    df = pd.DataFrame(out_rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Wording Check")
        ws = writer.sheets["Wording Check"]
        for excel_row, (is_non_standard, is_duplicate) in enumerate(row_flags, start=2):  # row 1 is the header
            for col_idx in range(1, len(df.columns) + 1):
                cell = ws.cell(row=excel_row, column=col_idx)
                if is_non_standard:
                    cell.font = NON_STANDARD_FONT
                if is_duplicate:
                    cell.fill = DUPLICATE_FILL

        class_list_lines = _build_class_list_lines(edited_df)
        pd.DataFrame({"": class_list_lines}).to_excel(
            writer, index=False, header=False, sheet_name="Class List"
        )
    return buf.getvalue()


def _build_class_list_lines(edited_df):
    """Groups the (possibly user-edited) rows by Class, using the final
    wording per item (Suggestion if present, else the original wording),
    formatted as ready-to-paste "Class N" / semicolon-list blocks."""
    groups = {}
    order = []
    for _, row in edited_df.fillna("").iterrows():
        cls = str(row.get("Class", "")).strip()
        suggestion = str(row.get("Suggestion", "")).strip()
        wording = str(row.get("Client wording", "")).strip()
        final_wording = (suggestion or wording).rstrip(";").strip()
        if not final_wording:
            continue
        if cls not in groups:
            groups[cls] = []
            order.append(cls)
        groups[cls].append(final_wording)

    lines = []
    for cls in order:
        lines.append(f"Class {cls}" if cls else "Class (unspecified)")
        lines.append("; ".join(groups[cls]) + ";")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def main():
    sidebar()
    check_page()


if __name__ == "__main__":
    main()
