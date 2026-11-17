"""Hybrid matching: whether an item's wording exactly matches a real
official term is a yes/no fact, checked for free against the downloaded
IPOS/MyIPO wording lists (see reference_match.py) — no need to pay an LLM
to answer a lookup. Only items with no exact match in the official list
("non-standard" candidates) are sent to Claude, which suggests the
corrected wording from its own knowledge of {authority} / NICE conventions,
since there's no official "closest correction" list to look up.
"""
import re

from . import reference_match
from .claude_client import authority_label, get_client

DEFAULT_MODEL = "claude-sonnet-5"
# Larger batches amortize the fixed system-prompt cost over more items per
# call. 25 items keeps output comfortably under max_tokens with margin.
DEFAULT_BATCH_SIZE = 25


def _normalize_literal(text):
    """Lowercase/whitespace-collapsed form used only to group exact-duplicate
    wording so Claude reviews each unique string once, not once per
    occurrence."""
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text.rstrip(";:.,")


_REFERENCE_INDEX_CACHE = {}


def _get_reference_index(authority):
    """Lazily builds and caches one ReferenceIndex per authority per process
    (building it is ~1s against MyIPO's 136k rows — worth doing once, not
    per request). Returns None if the reference data isn't available, so
    callers can fall back to sending everything to the agent rather than
    crashing the app over a missing/undeployed data file."""
    key = (authority or "").strip().lower()
    if key not in _REFERENCE_INDEX_CACHE:
        try:
            rows = reference_match.load_authority_reference(key)
            _REFERENCE_INDEX_CACHE[key] = reference_match.ReferenceIndex(rows)
        except Exception:
            _REFERENCE_INDEX_CACHE[key] = None
    return _REFERENCE_INDEX_CACHE[key]

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

IPOS (Singapore) and MyIPO (Malaysia) both use NICE classification as their base, but their accepted \
wording lists and formalities practice are not identical — a term accepted at one office is not automatically \
accepted at the other. You are reviewing specifically for {authority} in this session; apply that office's \
own conventions, not the other one's, even where they're very similar.

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
        max_tokens=6000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT.format(authority=authority_label(authority)),
            "cache_control": {"type": "ephemeral"},
        }],
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
    split_callback=None,
):
    """Returns one dict per input term, same shape/order as client_terms:
    {"input","status","best_score","suggestions":[{term,class,score}],"reasoning"}

    Stage 1 (free): each item is looked up against the official {authority}
    wording list (reference_match.py). An exact (class, wording) match, or a
    match under a different class, is resolved instantly with no API call —
    this is a yes/no fact, not a judgment call. If the reference data isn't
    available for some reason, every item falls through to stage 2 instead
    of crashing.

    Stage 2 (paid): only items with no exact match anywhere in the official
    list go to Claude for a suggested correction, batched with the existing
    exact-duplicate dedup + prompt caching.

    split_callback(free_count, agent_count), if given, is called once after
    stage 1 completes and before any Claude calls, so callers can show the
    user how many items were resolved for free.
    """
    classes = client_classes or [""] * len(client_terms)
    results = [None] * len(client_terms)
    ref_index = _get_reference_index(authority)

    agent_indices = []
    for i, (term, cls) in enumerate(zip(client_terms, classes)):
        if ref_index is not None:
            status, official_term, correct_class, is_compound = ref_index.lookup_compound(cls, term)
        else:
            status, official_term, correct_class, is_compound = "non_standard", None, None, False

        if status == "non_standard":
            agent_indices.append(i)
            continue

        # "standard_wrong_class" is still a free, certain match — just under
        # a different class than the client stated. Reporting the real class
        # here (not the client's) lets the existing cross-class-note logic
        # in streamlit_app.py flag the mismatch automatically, same as it
        # already does for a Claude-sourced suggestion.
        suggestion_class = correct_class if status == "standard_wrong_class" else cls
        reasoning = (
            f"Both halves of this compound line are separately standard in the official {authority.upper()} list."
            if is_compound else
            f"Exact match in the official {authority.upper()} wording list."
        )
        results[i] = {
            "input": term,
            "status": "standard",
            "best_score": None,
            "suggestions": [{"term": official_term, "class": suggestion_class, "score": None}],
            "reasoning": reasoning,
        }

    if split_callback:
        split_callback(len(client_terms) - len(agent_indices), len(agent_indices))

    agent_terms = [client_terms[i] for i in agent_indices]
    agent_classes = [classes[i] for i in agent_indices]
    agent_results = _review_nonstandard_with_claude(
        agent_terms, authority, agent_classes, api_key=api_key, model=model,
        batch_size=batch_size, progress_callback=progress_callback,
    )
    for local_i, global_i in enumerate(agent_indices):
        results[global_i] = agent_results[local_i]

    return results


def _review_nonstandard_with_claude(
    client_terms, authority, classes, api_key=None, model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE, progress_callback=None,
):
    """Sends the (already pre-filtered to non-standard) terms to Claude for
    a suggested correction, in batches. Exact-duplicate wording (case/
    whitespace/punctuation differences only) is reviewed once and the
    verdict copied to every matching occurrence — client lists often repeat
    the same item across several classes, and there's no reason to pay for
    the identical review twice."""
    results = [None] * len(client_terms)
    if not client_terms:
        return results

    groups = {}  # normalized text -> list of original indices
    unique_terms = []
    unique_classes = []
    for i, term in enumerate(client_terms):
        key = _normalize_literal(term)
        if key not in groups:
            groups[key] = []
            unique_terms.append(term)
            unique_classes.append(classes[i])
        groups[key].append(i)

    done = 0
    total_unique = len(unique_terms)
    for start in range(0, total_unique, batch_size):
        batch_terms = unique_terms[start:start + batch_size]
        batch_classes = unique_classes[start:start + batch_size]
        verdicts = review_batch(list(zip(batch_terms, batch_classes)), authority, api_key=api_key, model=model)
        verdict_by_input = {v["input"]: v for v in verdicts}

        for term in batch_terms:
            v = verdict_by_input.get(term)
            indices = groups[_normalize_literal(term)]

            if v is None:
                verdict_result = {
                    "status": "non_standard",
                    "best_score": None,
                    "suggestions": [],
                    "reasoning": "Claude did not return a verdict for this item.",
                }
            else:
                suggestion_term = (v.get("suggestion") or "").strip()
                suggestions = [{"term": suggestion_term, "class": v.get("class") or "", "score": None}] if suggestion_term else []

                for alt_term in v.get("alternatives") or []:
                    alt_term = (alt_term or "").strip()
                    if alt_term and alt_term != suggestion_term:
                        suggestions.append({"term": alt_term, "class": "", "score": None})

                verdict_result = {
                    "status": v.get("status") or "non_standard",
                    "best_score": None,
                    "suggestions": suggestions,
                    "reasoning": v.get("reasoning", ""),
                }

            for idx in indices:
                results[idx] = {"input": client_terms[idx], **verdict_result}

        done += len(batch_terms)
        if progress_callback:
            progress_callback(done, total_unique)

    return results
