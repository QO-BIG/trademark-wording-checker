"""Shared Anthropic client construction for claude_extractor.py and
claude_matcher.py."""
import os

import anthropic


def get_client(api_key=None):
    key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "No Anthropic API key available — set ANTHROPIC_API_KEY or enter a key in the sidebar."
        )
    if not key.startswith("sk-ant-"):
        raise RuntimeError(
            "This doesn't look like a valid Anthropic API key (it should start with 'sk-ant-'). "
            "Get one from console.anthropic.com → API Keys, and make sure no extra spaces/newlines "
            "were pasted along with it."
        )
    return anthropic.Anthropic(api_key=key)
