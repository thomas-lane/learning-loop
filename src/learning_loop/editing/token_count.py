"""Intervention-length accounting for counterfactual branch costs.

`intervention_tokens` is the learner-tokenizer length of the fixed assistant turn
exactly as it is rendered after the given prompt (same template, same
`chat_template_kwargs`, same completion boundary as DPO training), so an edited
action is never treated as free. The returned source string names the tokenizer
and template hashes, so counts from different tokenizers are never mixed silently.

`fixture_intervention_tokens` is a labeled estimate for scripted fixture runs only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

from ..core.config import ModelProfile, load_model_profile
from ..core.records import Message, ToolSchema

InterventionCounter = Callable[[list[Message], Message, list[ToolSchema]], tuple[int, str]]


def intervention_tokens(
    profile: ModelProfile | str,
    prompt_messages: list[Message],
    assistant_message: Message,
    tools: list[ToolSchema],
) -> tuple[int, str]:
    """(n_tokens, source). Raises training.render.RenderError on invalid messages."""
    from ..training.render import chat_template_sha256, end_of_turn_ids, load_tokenizer, render_completion, tokenizer_sha256

    prof = load_model_profile(profile) if isinstance(profile, str) else profile
    tok = load_tokenizer(prof)
    rc = render_completion(
        tok,
        prompt_messages,
        [assistant_message],
        tools,
        chat_template_kwargs=prof.chat_template_kwargs,
        eot_ids=end_of_turn_ids(tok, prof),
    )
    source = f"hf_template:{prof.name}:tokenizer={tokenizer_sha256(tok)[:16]}:template={chat_template_sha256(tok)[:16]}"
    return len(rc.completion_ids), source


def fixture_intervention_tokens(
    prompt_messages: list[Message], assistant_message: Message, tools: list[ToolSchema]
) -> tuple[int, str]:
    """ceil(chars/4) of the canonical JSON of the assistant turn. Fixture runs only."""
    payload = {
        "content": assistant_message.get("content") or "",
        "tool_calls": [
            {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments")}
            for tc in assistant_message.get("tool_calls") or []
        ],
    }
    n = math.ceil(len(json.dumps(payload, sort_keys=True, separators=(",", ":"))) / 4)
    return n, "fixture_estimate:ceil(chars/4)"


def get_counter(profile: ModelProfile | str | None) -> InterventionCounter:
    """Counter for the verifier: learner tokenizer when a profile is given, else the fixture estimate."""
    if profile is None:
        return fixture_intervention_tokens
    prof = load_model_profile(profile) if isinstance(profile, str) else profile

    def count(prompt: list[Message], msg: Message, tools: list[ToolSchema]) -> tuple[int, str]:
        return intervention_tokens(prof, prompt, msg, tools)

    return count


__all__: list[Any] = ["intervention_tokens", "fixture_intervention_tokens", "get_counter", "InterventionCounter"]
