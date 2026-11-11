"""Claude-reviewed matching: Claude judges each item's standard/non-standard
status and suggests the corrected wording directly from its own knowledge
of {authority}'s official classification conventions — the same as pasting
the client's wording into a Claude chat and asking it to check it. There is
no local reference list or retrieval step; nothing here is grounded in a
downloaded, verifiable official list, which is a deliberate tradeoff for
simplicity over verifiability.
"""
from .claude_client import get_client

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_BATCH_SIZE = 15

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
                            "description": "The official wording this item corresponds to, to the best of your knowledge — always provide this: if status is standard, the official term the wording already matches; if non_standard, the corrected wording to use instead. If the client's single line actually names more than one distinct accepted concept, include each matching official term here separated by '; ' — do not silently drop one of them for a single-term answer. Empty string ONLY if you genuinely don't know a correct official term for this.",
                        },
                        "class": {
                            "type": "string",
                            "description": "NICE class this wording belongs under, to the best of your knowledge. Empty string only if suggestion is empty.",
                        },
                        "alternatives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Other official terms that could also reasonably apply, when genuinely ambiguous between two plausible interpretations of the client's wording. Leave empty when there is one clearly best answer; don't pad this with weaker options just to fill it.",
                        },
                        "reasoning": {"type": "string", "description": "One short sentence explaining the verdict."},
                    },
                    "required": ["input", "status", "suggestion", "class", "alternatives", "reasoning"],
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

For each item you are given the client's requested wording (and, when known, the NICE class the client \
filed it under — this may not be the class it's actually correctly classified in).

Apply that professional judgment, per item, drawing on your own knowledge of accepted {authority} / NICE \
classification wording — there is no reference list provided here, so rely on what you know to be genuinely \
standard, accepted terminology rather than guessing at something plausible-sounding:
- "standard": the client's wording already matches a real official term (differences limited to case, \
punctuation, or whitespace only). Still report that matching official term as the suggestion — every item \
should show the official wording it corresponds to, not just the ones that need correcting.
- "non_standard": anything an examiner would require amending — a synonym, non-standard business phrasing \
(e.g. "online retail store services" instead of "retail services"), a typo, wrong specificity, outdated \
terminology, etc. Give the best-fitting real official term(s) as the suggestion. If you don't genuinely know \
a correct official term for this, leave suggestion empty and say why in reasoning — never invent something \
that merely sounds plausible.
- If the client's single line actually names more than one distinct accepted concept (e.g. "glue and \
adhesives for industrial purposes" really covers two separate official terms, "Glues for industrial \
purposes" and "Adhesives for industrial purposes"), include every matching official term in the \
suggestion, separated by "; " — never silently drop one of the concepts just to give a single-term answer.
- If the correct classification for this wording is a different NICE class than the client specified, still \
give that correct term as the suggestion — don't discard a good match over its class label — and say so \
explicitly in reasoning (e.g. "Standard wording, but properly classified under Class 35 rather than the \
stated Class 42").

Return exactly one result per item, using the item's "input" text verbatim so results can be matched back up."""


def _format_batch_prompt(batch):
    lines = []
    for i, (term, cls) in enumerate(batch, 1):
        lines.append(f"Item {i}:")
        lines.append(f'  Client wording: "{term}"')
        if cls:
            lines.append(f"  Client-specified NICE class: {cls}")
        lines.append("")
    return "\n".join(lines)


def review_batch(batch, authority, api_key=None, model=DEFAULT_MODEL):
    """batch: list of (term, client_class) tuples."""
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
    client_classes=None,
    api_key=None,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
):
    """Sends client terms directly to Claude for review in batches — no
    local reference list, no retrieval step. Returns one dict per input
    term: {"input","status","best_score","suggestions":[{term,class,score}],"reasoning"}
    """
    classes = client_classes or [""] * len(client_terms)
    results = [None] * len(client_terms)

    done = 0
    for start in range(0, len(client_terms), batch_size):
        batch_terms = client_terms[start:start + batch_size]
        batch_classes = classes[start:start + batch_size]
        verdicts = review_batch(list(zip(batch_terms, batch_classes)), authority, api_key=api_key, model=model)
        verdict_by_input = {v["input"]: v for v in verdicts}

        for i, term in enumerate(batch_terms):
            global_i = start + i
            v = verdict_by_input.get(term)

            if v is None:
                results[global_i] = {
                    "input": term,
                    "status": "non_standard",
                    "best_score": None,
                    "suggestions": [],
                    "reasoning": "Claude did not return a verdict for this item.",
                }
                continue

            suggestion_term = (v.get("suggestion") or "").strip()
            suggestions = [{"term": suggestion_term, "class": v.get("class") or "", "score": None}] if suggestion_term else []

            for alt_term in v.get("alternatives") or []:
                alt_term = (alt_term or "").strip()
                if alt_term and alt_term != suggestion_term:
                    suggestions.append({"term": alt_term, "class": "", "score": None})

            results[global_i] = {
                "input": term,
                "status": v.get("status") or "non_standard",
                "best_score": None,
                "suggestions": suggestions,
                "reasoning": v.get("reasoning", ""),
            }

        done += len(batch_terms)
        if progress_callback:
            progress_callback(done, len(client_terms))

    return results
