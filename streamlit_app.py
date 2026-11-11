"""Streamlit app:
  - Check client wording upload a client's wording and get Claude-reviewed
                          suggestions based on the currently loaded list
  - Upload official list refresh the list when IPOS/MyIPO publish an update

Run with: streamlit run streamlit_app.py
"""
import io
import os

import pandas as pd
import streamlit as st
from openpyxl.styles import Font, PatternFill

from src import claude_extractor, claude_matcher, matcher, parsers, reference_store

st.set_page_config(page_title="Trademark Wording Checker", page_icon="⚖️", layout="wide")

LARGE_ROW_THRESHOLD = 1000


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
    page = st.sidebar.radio("Go to", ["Check client wording", "Upload official list"])

    st.sidebar.divider()
    if get_api_key():
        st.sidebar.success("Anthropic API key configured.")
    else:
        st.sidebar.warning(
            "No Anthropic API key set. Add ANTHROPIC_API_KEY to .streamlit/secrets.toml to enable Claude."
        )
    return page


def _guess_cols_ui(columns, guessed_class_col, guessed_term_col, key_prefix, uniq):
    col1, col2 = st.columns(2)
    class_options = ["(none)"] + columns
    class_index = class_options.index(guessed_class_col) if guessed_class_col in columns else 0
    term_index = columns.index(guessed_term_col) if guessed_term_col in columns else 0

    with col1:
        class_col = st.selectbox(
            "Class column (optional)", class_options, index=class_index, key=f"{key_prefix}_class_col_{uniq}"
        )
    with col2:
        term_col = st.selectbox(
            "Wording column", columns, index=term_index, key=f"{key_prefix}_term_col_{uniq}"
        )

    return (None if class_col == "(none)" else class_col), term_col


def upload_page():
    st.title("Upload official wording list")
    st.caption("Use this whenever IPOS or MyIPO publishes an updated pre-approved goods/services list.")

    authority = st.selectbox("Authority", ["ipos", "myipo"], format_func=str.upper, key="upload_authority_select")
    uploaded = st.file_uploader("File", type=["xlsx", "xls", "csv", "pdf", "docx"], key="upload_file")

    if uploaded is None:
        return

    cache_key = (uploaded.name, uploaded.size, authority)
    if st.session_state.get("upload_cache_key") != cache_key:
        with st.spinner(f"Parsing {uploaded.name}..."):
            st.session_state["upload_parsed"] = parsers.parse_upload(uploaded.getvalue(), uploaded.name)
        st.session_state["upload_cache_key"] = cache_key
        st.session_state["upload_filename"] = uploaded.name

    parsed = st.session_state["upload_parsed"]
    filename = st.session_state["upload_filename"]
    uniq = f"{filename}_{uploaded.size}"

    if parsed["kind"] == "tabular":
        class_col, term_col = _guess_cols_ui(
            parsed["columns"], parsed["guessed_class_col"], parsed["guessed_term_col"], "upload", uniq
        )
        rows = parsers.rows_from_tabular(parsed["all_rows"], class_col, term_col)
    else:
        rows = parsed["all_rows"]

    st.write(f"**{len(rows):,}** terms detected for **{authority.upper()}** from {filename}.")

    if len(rows) > LARGE_ROW_THRESHOLD:
        st.info("List is large — showing a preview only. Check the sample looks right, then confirm to save the full list.")
        st.dataframe(pd.DataFrame(rows[:30]), hide_index=True, width="stretch")
        if st.button(f"Save all {len(rows):,} terms as new {authority.upper()} version", type="primary"):
            version_id = reference_store.save_new_version(authority, rows, filename)
            st.success(f"{authority.upper()} list updated: version {version_id} with {len(rows):,} terms loaded.")
    else:
        st.caption("Edit any row below to fix misparsed entries before saving.")
        edited_df = st.data_editor(
            pd.DataFrame(rows), num_rows="dynamic", width="stretch", key=f"upload_editor_{uniq}"
        )
        if st.button(f"Save {len(edited_df)} terms as new {authority.upper()} version", type="primary"):
            final_rows = [r for r in edited_df.fillna("").to_dict("records") if str(r.get("term", "")).strip()]
            version_id = reference_store.save_new_version(authority, final_rows, filename)
            st.success(f"{authority.upper()} list updated: version {version_id} with {len(final_rows)} terms loaded.")


def check_page():
    st.title("Check client wording")
    st.caption("Upload the client's requested goods/services wording to check it against the current official list.")

    summary = reference_store.status_summary()
    status_cols = st.columns(len(summary))
    for col, (a, info) in zip(status_cols, summary.items()):
        col.metric(a.upper(), f"{info['term_count']:,} terms", info["current_version"] or "none loaded")

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
                terms, authority, api_key=api_key, progress_callback=on_progress
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
        key = matcher.normalize_literal(str(r.get("input", "")))
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
    no_reference = any(r["status"] == "no_reference_loaded" for r in results)

    st.subheader("Results")
    if no_reference:
        st.error(f"No {data['authority'].upper()} list is loaded yet. Upload one first.")
        return

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

    st.caption("Non-standard rows in red, exact duplicates highlighted (read-only view):")
    st.dataframe(
        pd.DataFrame(display_rows).style.apply(highlight_row, axis=1),
        hide_index=True,
        width="stretch",
    )

    st.caption("Edit any field below before downloading the report.")
    edited_df = st.data_editor(
        pd.DataFrame(display_rows),
        num_rows="dynamic",
        width="stretch",
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
            "Notes": str(row.get("Reasoning", "")).strip(),
        })
        last_class = cls
        row_flags.append((
            str(row.get("Status", "")).strip() == "Non-standard",
            str(row.get("Duplicate", "")).strip() == "Yes",
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
    return buf.getvalue()


def main():
    page = sidebar()
    if page == "Upload official list":
        upload_page()
    elif page == "Check client wording":
        check_page()


if __name__ == "__main__":
    main()
