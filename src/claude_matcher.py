"""Claude-reviewed matching: the local TF-IDF/fuzzy index (matcher.py)
shortlists candidate official terms, then Claude reviews the client's
wording against that shortlist and makes the actual standard/non-standard
call and picks the best replacement wording.

Candidates are shortlisted locally because the reference lists run to
100k+ rows; sending the full list to Claude on every check isn't feasible.
"""
from . import matcher
from .claude_client import get_client

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_BATCH_SIZE = 8

REVIEW_TOOL = {
    "name": "report_wording_review",
    "description": "Report the review verdict for each client wording item in this batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The client wording text being reviewed, copied verbatim from the prompt.",
                        },
                        "status": {"type": "string", "enum": ["standard", "non_standard"]},
                        "suggestion": {
                            "type": "string",
                            "description": "The official wording this item corresponds to, copied VERBATIM from this item's candidate list — always provide this: if status is standard, the official term the wording already matches; if non_standard, the corrected wording to use instead. If the client's single line actually names more than one distinct accepted concept (e.g. 'glue and adhesives for industrial purposes' covering two separate official terms), include each matching official term here separated by '; ' — do not silently drop one of them for a single-term answer. Empty string ONLY if status is non_standard and no candidate genuinely fits.",
                        },
                        "class": {
                            "type": "string",
                            "description": "NICE class of the suggestion, copied from the candidate list. Empty string only if suggestion is empty.",
                        },
                        "reasoning": {"type": "string", "description": "One short sentence explaining the verdict."},
                    },
                    "required": ["input", "status", "suggestion", "class", "reasoning"],
                },
            }
        },
        "required": ["items"],
    },
}

SYSTEM_PROMPT = """You are an experienced trademark prosecution agent handling {authority} filings, with \
deep working knowledge of the NICE International Classification of Goods and Services and of how {authority} \
examiners actually assess goods/services wording in practice — what gets accepted at formalities \
examination versus what draws an objection or amendment request: overly broad or vague wording, marketing-\
style business jargon instead of accepted legal phrasing, outdated nomenclature, and wrong specificity \
(too broad or too narrow for any accepted term) are all things a real examiner would flag, not just a \
literal mismatch.

For each item you are given the client's requested wording and a shortlist of the closest official terms \
(already retrieved by a local search index — the right answer may or may not be among them, and the \
shortlist can include terms from a NICE class other than what the client specified, since a term's \
correctly classified home can differ from what the client assumed).

Apply that professional judgment, per item:
- "standard": the client's wording already matches an official term (differences limited to case, \
punctuation, or whitespace only). Still report that matching official term as the suggestion, copied \
VERBATIM from the candidate list — every item should show the official wording it corresponds to, not \
just the ones that need correcting.
- "non_standard": anything an examiner would require amending — a synonym, non-standard business phrasing \
(e.g. "online retail store services" instead of "retail services"), a typo, wrong specificity, outdated \
terminology, etc. Pick the best-fitting official term(s) from THIS ITEM'S candidate list as the suggestion, \
copied VERBATIM (exact spelling/wording) from that list. If none of the candidates genuinely capture the \
intended meaning, leave suggestion empty and say why in reasoning.
- If the client's single line actually names more than one distinct accepted concept (e.g. "glue and \
adhesives for industrial purposes" really covers two separate official terms, "Glues for industrial \
purposes" and "Adhesives for industrial purposes"), include every matching official term in the \
suggestion, separated by "; " — never silently drop one of the concepts just to give a single-term answer.
- When multiple candidates are each plausible and you must pick one, prefer whichever shares the most \
literal key nouns/words with the client's wording (e.g. for "priming putty", a candidate containing the \
word "putty" is a stronger match than one that only shares "priming") — don't let a looser semantic \
association outrank a closer literal match.
- If the best-fitting candidate belongs to a different NICE class than the client specified for this item, \
still select it as the suggestion — don't discard a good match over its class label — and say so explicitly \
in reasoning (e.g. "Standard wording, but properly classified under Class 35 rather than the stated Class 42").

Never invent wording that is not present in the item's own candidate list. Never swap in a candidate from \
a different item. Return exactly one result per item, using the item's "input" text verbatim so results can \
be matched back up."""


def _format_batch_prompt(batch):
    lines = []
    for i, item in enumerate(batch, 1):
        lines.append(f"Item {i}:")
        lines.append(f'  Client wording: "{item["input"]}"')
        if item["candidates"]:
            lines.append("  Candidate official terms:")
            for c in item["candidates"]:
                lines.append(f'    - "{c["term"]}" (Class {c["class"]})')
        else:
            lines.append("  Candidate official terms: (none found by the local search)")
        lines.append("")
    return "\n".join(lines)


def review_batch(batch, authority, api_key=None, model=DEFAULT_MODEL):
    """batch: list of {"input", "candidates":[{"term","class","score"}]} —
    items that already failed the local exact-match check."""
    if not batch:
        return []

    client = get_client(api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT.format(authority=authority.upper()),
        tools=[REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "report_wording_review"},
        messages=[{"role": "user", "content": _format_batch_prompt(batch)}],
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "report_wording_review":
            return block.input.get("items", [])
    return []


def match_terms_with_claude(
    client_terms,
    authority,
    api_key=None,
    model=DEFAULT_MODEL,
    top_k_candidates=15,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
):
    """Full pipeline: local shortlist, then Claude review for everything
    that isn't a literal exact match. Returns one dict per input term:
      {"input","status","best_score","suggestions":[{term,class,score}],"reasoning"}
    """
    reference, prepared = matcher.prepare_candidates(client_terms, authority, top_k=top_k_candidates)
    if reference is None:
        return [
            {"input": t, "status": "no_reference_loaded", "best_score": None, "suggestions": [], "reasoning": ""}
            for t in client_terms
        ]

    results = [None] * len(prepared)
    to_review, review_indices = [], []

    for i, item in enumerate(prepared):
        if item["exact_match"]:
            results[i] = {
                "input": item["input"],
                "status": "standard",
                "best_score": 1.0,
                "suggestions": [{**item["exact_match"], "score": 1.0}],
                "reasoning": "Exact match to official wording.",
            }
        else:
            to_review.append(item)
            review_indices.append(i)

    done = 0
    for start in range(0, len(to_review), batch_size):
        batch = to_review[start:start + batch_size]
        verdicts = review_batch(batch, authority, api_key=api_key, model=model)
        verdict_by_input = {v["input"]: v for v in verdicts}

        for local_i, item in enumerate(batch):
            global_i = review_indices[start + local_i]
            v = verdict_by_input.get(item["input"])

            if v is None:
                top = item["candidates"][0] if item["candidates"] else None
                results[global_i] = {
                    "input": item["input"],
                    "status": "non_standard",
                    "best_score": top["score"] if top else None,
                    "suggestions": [top] if top else [],
                    "reasoning": "Claude did not return a verdict for this item; showing the closest local match instead.",
                }
                continue

            suggestion_term = (v.get("suggestion") or "").strip()
            suggestions = []
            if suggestion_term:
                matching_candidate = next(
                    (c for c in item["candidates"] if c["term"] == suggestion_term), None
                )
                suggestions = [{
                    "term": suggestion_term,
                    "class": v.get("class") or (matching_candidate["class"] if matching_candidate else ""),
                    "score": matching_candidate["score"] if matching_candidate else None,
                }]

            results[global_i] = {
                "input": item["input"],
                "status": v.get("status") or "non_standard",
                "best_score": suggestions[0]["score"] if suggestions else None,
                "suggestions": suggestions,
                "reasoning": v.get("reasoning", ""),
            }

        done += len(batch)
        if progress_callback:
            progress_callback(done, len(to_review))

    return results
