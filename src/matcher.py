"""Local shortlisting of candidate official terms from the current
reference list (IPOS or MyIPO), used by claude_matcher.py to narrow 100k+
rows down to a small set worth sending to Claude for review.

Approach:
  1. a small curated synonym list for common trademark-wording variants
     (e.g. "hand phone" -> "mobile phone"), applied before comparison
  2. TF-IDF over character n-grams + word n-grams, which is tolerant of
     reordering, pluralization and minor phrasing differences
  3. a fuzzy string score (rapidfuzz) blended in as a second signal
This is deliberately cheap/local — it only needs to be good enough to put
the right answer somewhere in the shortlist; Claude makes the final call.
"""
import pickle

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import reference_store

# Starter list of common trademark-wording variants seen in SG/MY filings.
# Extend this as you notice recurring client phrasing that should map to a
# specific official term.
SYNONYMS = {
    "hand phone": "mobile phone",
    "handphone": "mobile phone",
    "cell phone": "mobile phone",
    "cellphone": "mobile phone",
    "cellular phone": "mobile phone",
    "cellular telephone": "mobile phone",
    "smart phone": "mobile phone",
    "smartphone": "mobile phone",
    "apparel": "clothing",
    "garments": "clothing",
    "shoes": "footwear",
    "online retail store services": "retail services",
    "e-commerce services": "retail services",
    "f&b": "food and beverage",
}

_INDEX_CACHE = {}  # (authority, version_id) -> (vectorizer, matrix)


def normalize_literal(text):
    """Lowercase/whitespace/trailing-punctuation normalization only — used
    for the exact-match check, so e.g. a trailing ';' doesn't turn an
    already-standard term into a false "non-standard" result, but two
    genuinely different synonyms are never silently treated as identical.
    """
    t = " ".join(text.strip().lower().split())
    return t.rstrip(";:.,")


def _normalize_for_similarity(text):
    """Literal normalization plus synonym substitution — used only to rank
    candidates, never to decide whether something already counts as an
    exact, standard match.
    """
    t = normalize_literal(text)
    for phrase, canonical in SYNONYMS.items():
        if phrase in t:
            t = t.replace(phrase, canonical)
    return t


def _index_path(authority, version_id):
    return reference_store.DATA_DIR / authority / "versions" / f"{version_id}.tfidf.pkl"


def _build_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)


def _get_index(authority, version_id, ref_terms_normalized):
    key = (authority, version_id)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    path = _index_path(authority, version_id)
    if path.exists():
        with open(path, "rb") as f:
            vectorizer, matrix = pickle.load(f)
    else:
        vectorizer = _build_vectorizer()
        matrix = vectorizer.fit_transform(ref_terms_normalized)
        with open(path, "wb") as f:
            pickle.dump((vectorizer, matrix), f)

    _INDEX_CACHE[key] = (vectorizer, matrix)
    return vectorizer, matrix


def prepare_candidates(client_terms, authority, top_k=15):
    """Shortlist step shared by the local matcher and the Claude-reviewed
    matcher: for each client term, either it's a literal exact match to an
    official term (cheap, confident, no need to spend an API call on it),
    or it gets a ranked shortlist of the top_k closest official terms by
    TF-IDF + fuzzy similarity for a smarter reviewer (e.g. Claude) to pick
    from, since sending the full 100k+ term list per query isn't feasible.

    Returns (reference_or_None, list of per-term dicts):
      {"input": str, "exact_match": {"term","class"} | None,
       "candidates": [{"term","class","score"}, ...]}
    """
    reference = reference_store.load_current(authority)
    if reference is None or not reference["rows"]:
        return None, [{"input": t, "exact_match": None, "candidates": []} for t in client_terms]

    rows = reference["rows"]
    ref_terms = [r["term"] for r in rows]
    ref_terms_literal = [normalize_literal(t) for t in ref_terms]
    ref_terms_for_similarity = [_normalize_for_similarity(t) for t in ref_terms]
    literal_lookup = {t: i for i, t in enumerate(ref_terms_literal)}

    vectorizer, ref_matrix = _get_index(authority, reference["version_id"], ref_terms_for_similarity)

    out = [None] * len(client_terms)
    pending_indices, pending_normalized = [], []

    for i, raw_term in enumerate(client_terms):
        literal = normalize_literal(raw_term)
        if literal in literal_lookup:
            idx = literal_lookup[literal]
            out[i] = {
                "input": raw_term,
                "exact_match": {"term": rows[idx]["term"], "class": rows[idx]["class"]},
                "candidates": [],
            }
        else:
            pending_indices.append(i)
            pending_normalized.append(_normalize_for_similarity(raw_term))

    if pending_normalized:
        query_matrix = vectorizer.transform(pending_normalized)
        sims = cosine_similarity(query_matrix, ref_matrix)

        for local_i, global_i in enumerate(pending_indices):
            norm_term = pending_normalized[local_i]
            sim_row = sims[local_i]
            fuzzy_scores = np.array([
                fuzz.token_sort_ratio(norm_term, t) / 100.0 for t in ref_terms_for_similarity
            ])
            blended = 0.6 * sim_row + 0.4 * fuzzy_scores
            top_idx = np.argsort(-blended)[:top_k]
            candidates = [
                {"term": ref_terms[idx], "class": rows[idx]["class"], "score": round(float(blended[idx]), 4)}
                for idx in top_idx
            ]
            out[global_i] = {
                "input": client_terms[global_i],
                "exact_match": None,
                "candidates": candidates,
            }

    return reference, out
