"""Render preference examples and conversations with the learner's own chat template.

One rendering path is shared by training (DPO rows), intervention-token
accounting (`token_count.py`) and the reference HF server, so the tokens the
learner is trained on are the tokens it is served with.

Conventions (tested in tests/unit/test_train_render.py):

- Records hold tool-call `arguments` as the JSON *string* that entered the
  history. Templates are given a parsed JSON *object* instead, so every template
  serializes arguments its own way (Qwen3: `tojson`; Gemma 4: sorted
  `key:<|"|>value<|"|>`), identically for chosen, rejected and prompt turns,
  regardless of the spacing the endpoint happened to emit.
- prompt     = template(prompt messages, add_generation_prompt=True)
- completion = template(prompt + [assistant turn])[len(prompt):], cut right
  after the first end-of-turn token (tokenizer EOS / generation-config EOS ids),
  so trailing template whitespace the model never generates is not trained on.
- The prompt token ids must be an exact prefix of both full renders; otherwise
  the template is unusable for DPO and rendering fails loudly.
- Oversize examples are dropped with a recorded reason, never truncated.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelProfile, load_model_profile
from ..records import Message, PreferenceExample, ToolSchema


class RenderError(ValueError):
    """The example or template violates the rendering contract (not a data-size issue)."""


# --------------------------------------------------------------------------- #
# Tokenizer identity
# --------------------------------------------------------------------------- #


def tokenizer_source(profile: ModelProfile) -> tuple[str, str]:
    """(repo, revision) of the tokenizer. `tokenizer: repo@sha` overrides base_model@base_revision."""
    if profile.tokenizer:
        repo, _, rev = profile.tokenizer.partition("@")
        return repo, rev or profile.base_revision
    return profile.base_model, profile.base_revision


@functools.lru_cache(maxsize=8)
def _load_tokenizer(repo: str, revision: str, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=local_files_only)


def load_tokenizer(profile: ModelProfile, local_files_only: bool | None = None, verify_template: bool = True) -> Any:
    """Load the pinned tokenizer (cache first, then hub) and verify the pinned template hash."""
    repo, rev = tokenizer_source(profile)
    if local_files_only is None:
        try:
            tok = _load_tokenizer(repo, rev, True)
        except OSError:
            tok = _load_tokenizer(repo, rev, False)
    else:
        tok = _load_tokenizer(repo, rev, local_files_only)
    if verify_template and profile.chat_template_sha256:
        got = chat_template_sha256(tok)
        if got != profile.chat_template_sha256:
            raise RenderError(
                f"{profile.name}: chat template sha256 {got} != pinned {profile.chat_template_sha256}; "
                "the tokenizer/template changed — re-pin the profile deliberately"
            )
    return tok


def chat_template_sha256(tok: Any) -> str:
    tmpl = tok.chat_template
    if not isinstance(tmpl, str):
        tmpl = json.dumps(tmpl, sort_keys=True)
    return hashlib.sha256(tmpl.encode()).hexdigest()


_TOK_SHA: dict[int, tuple[Any, str]] = {}


def tokenizer_sha256(tok: Any) -> str:
    """Hash of the full tokenizer definition (vocab, merges, normalizers, added tokens)."""
    hit = _TOK_SHA.get(id(tok))
    if hit is not None and hit[0] is tok:
        return hit[1]
    sha = _tokenizer_sha256_uncached(tok)
    _TOK_SHA[id(tok)] = (tok, sha)
    return sha


def _tokenizer_sha256_uncached(tok: Any) -> str:
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        payload = backend.to_str()
    else:  # slow tokenizer: vocab + special tokens
        payload = json.dumps(tok.get_vocab(), sort_keys=True)
    extra = json.dumps({"special": tok.special_tokens_map, "cls": type(tok).__name__}, sort_keys=True, default=str)
    return hashlib.sha256((payload + "\n" + extra).encode()).hexdigest()


def end_of_turn_ids(tok: Any, profile: ModelProfile | None = None) -> set[int]:
    """Tokens that end a generated assistant turn: tokenizer EOS + generation-config EOS ids."""
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    if profile is not None:
        ids |= _generation_eos(*tokenizer_source(profile))
    return ids


@functools.lru_cache(maxsize=8)
def _generation_eos(repo: str, revision: str) -> frozenset[int]:
    from transformers import GenerationConfig

    try:
        gc = GenerationConfig.from_pretrained(repo, revision=revision, local_files_only=True)
    except OSError:
        try:
            gc = GenerationConfig.from_pretrained(repo, revision=revision)
        except OSError:
            return frozenset()
    eos = gc.eos_token_id
    if eos is None:
        return frozenset()
    return frozenset([eos] if isinstance(eos, int) else [int(e) for e in eos])


# --------------------------------------------------------------------------- #
# Messages -> template inputs
# --------------------------------------------------------------------------- #


def parse_arguments(raw: Any, where: str) -> dict[str, Any]:
    """History arguments are JSON strings; templates want an object. Invalid -> RenderError."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise RenderError(f"{where}: tool-call arguments must be a JSON string, got {type(raw).__name__}")
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RenderError(f"{where}: tool-call arguments are not valid JSON: {e}") from e
    if not isinstance(val, dict):
        raise RenderError(f"{where}: tool-call arguments must be a JSON object, got {type(val).__name__}")
    return val


def to_template_messages(messages: list[Message]) -> list[Message]:
    """Copy of `messages` with tool-call arguments parsed and `content: None` -> ""."""
    out: list[Message] = []
    for i, m in enumerate(messages):
        m = copy.deepcopy(m)
        if m.get("content") is None:
            m["content"] = ""
        for j, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            fn["arguments"] = parse_arguments(fn.get("arguments"), f"message {i} tool_call {j}")
            tc["function"] = fn
        out.append(m)
    return out


def render_text(
    tok: Any,
    messages: list[Message],
    tools: list[ToolSchema] | None,
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    return tok.apply_chat_template(
        to_template_messages(messages),
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **(chat_template_kwargs or {}),
    )


def encode(tok: Any, text: str) -> list[int]:
    """Tokenize rendered template text. Templates insert their own BOS, so no special tokens."""
    return list(tok(text, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------- #
# Completions
# --------------------------------------------------------------------------- #


@dataclass
class RenderedCompletion:
    prompt_text: str
    prompt_ids: list[int]
    completion_text: str  # decoded completion ids (through the end-of-turn token)
    completion_ids: list[int]
    trailing_dropped: list[int]  # template tokens after end-of-turn that are not trained on


def render_completion(
    tok: Any,
    prompt: list[Message],
    completion: list[Message],
    tools: list[ToolSchema] | None,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    eot_ids: set[int],
    prompt_text: str | None = None,
    prompt_ids: list[int] | None = None,
) -> RenderedCompletion:
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise RenderError("completion must be exactly one assistant message")
    if prompt_text is None:
        prompt_text = render_text(tok, prompt, tools, add_generation_prompt=True, chat_template_kwargs=chat_template_kwargs)
    if prompt_ids is None:
        prompt_ids = encode(tok, prompt_text)
    full_text = render_text(tok, prompt + completion, tools, add_generation_prompt=False, chat_template_kwargs=chat_template_kwargs)
    if not full_text.startswith(prompt_text):
        raise RenderError("rendered prompt text is not a prefix of the rendered prompt+completion")
    full_ids = encode(tok, full_text)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RenderError("prompt token ids are not an exact prefix of prompt+completion token ids")
    rest = full_ids[len(prompt_ids) :]
    cut = next((k for k, t in enumerate(rest) if t in eot_ids), None)
    if cut is None:
        raise RenderError("rendered completion has no end-of-turn token")
    ids = rest[: cut + 1]
    return RenderedCompletion(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        completion_text=tok.decode(ids, skip_special_tokens=False),
        completion_ids=ids,
        trailing_dropped=rest[cut + 1 :],
    )


# --------------------------------------------------------------------------- #
# Preference pairs
# --------------------------------------------------------------------------- #


def example_sha256(ex: PreferenceExample) -> str:
    payload = {"prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected, "tools": ex.tools}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class RenderedPair:
    pair_id: str
    example_sha256: str
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]
    prompt_text: str
    chosen_text: str
    rejected_text: str

    @property
    def token_ids_sha256(self) -> str:
        blob = json.dumps([self.prompt_ids, self.chosen_ids, self.rejected_ids], separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def total_length(self) -> int:
        return len(self.prompt_ids) + max(len(self.chosen_ids), len(self.rejected_ids))

    def labels(self, which: str) -> list[int]:
        """Causal-LM labels: -100 on every prompt position (tool observations included)."""
        comp = self.chosen_ids if which == "chosen" else self.rejected_ids
        return [-100] * len(self.prompt_ids) + list(comp)

    def row(self) -> dict[str, Any]:
        return {"prompt_ids": self.prompt_ids, "chosen_ids": self.chosen_ids, "rejected_ids": self.rejected_ids}


@dataclass
class RenderReport:
    kept: list[RenderedPair] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)  # [{pair_id, reason}]

    max_length: int | None = None

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for d in self.dropped:
            key = d["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        lengths = sorted(p.total_length for p in self.kept)
        return {
            "n_input": len(self.kept) + len(self.dropped),
            "n_kept": len(self.kept),
            "n_dropped": len(self.dropped),
            "dropped_by_reason": reasons,
            "dropped": self.dropped,
            "max_length": self.max_length,
            # prompt + longer completion, in tokens, of the pairs actually trained on
            "kept_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]} if lengths else None,
        }


def render_pair(
    ex: PreferenceExample,
    tok: Any,
    *,
    chat_template_kwargs: dict[str, Any] | None,
    eot_ids: set[int],
    max_length: int | None,
) -> RenderedPair | str:
    """Render one pair; returns a drop reason string for data-dependent exclusions."""
    for label, msgs in (("chosen", ex.chosen), ("rejected", ex.rejected)):
        if len(msgs) != 1 or msgs[0].get("role") != "assistant":
            return f"invalid_{label}:must be exactly one assistant message"
        if len(msgs[0].get("tool_calls") or []) != 1:
            return f"invalid_{label}:must contain exactly one tool call"
    if not ex.prompt or ex.prompt[-1].get("role") == "assistant":
        return "invalid_prompt:empty or ends with an assistant message"
    try:
        prompt_text = render_text(tok, ex.prompt, ex.tools, add_generation_prompt=True, chat_template_kwargs=chat_template_kwargs)
        prompt_ids = encode(tok, prompt_text)
        ch = render_completion(
            tok, ex.prompt, ex.chosen, ex.tools, chat_template_kwargs=chat_template_kwargs, eot_ids=eot_ids,
            prompt_text=prompt_text, prompt_ids=prompt_ids,
        )
        rj = render_completion(
            tok, ex.prompt, ex.rejected, ex.tools, chat_template_kwargs=chat_template_kwargs, eot_ids=eot_ids,
            prompt_text=prompt_text, prompt_ids=prompt_ids,
        )
    except RenderError as e:
        if "arguments" in str(e):
            return f"invalid_arguments:{e}"
        raise
    if ch.completion_ids == rj.completion_ids:
        return "chosen_equals_rejected:identical completion tokens"
    pair = RenderedPair(
        pair_id=ex.pair_id,
        example_sha256=example_sha256(ex),
        prompt_ids=prompt_ids,
        chosen_ids=ch.completion_ids,
        rejected_ids=rj.completion_ids,
        prompt_text=prompt_text,
        chosen_text=ch.completion_text,
        rejected_text=rj.completion_text,
    )
    if max_length is not None and pair.total_length > max_length:
        return f"oversize:{pair.total_length}>{max_length}"
    return pair


def render_pairs(
    examples: list[PreferenceExample],
    tok: Any,
    *,
    chat_template_kwargs: dict[str, Any] | None,
    eot_ids: set[int],
    max_length: int | None,
) -> RenderReport:
    rep = RenderReport(max_length=max_length)
    for ex in examples:
        out = render_pair(ex, tok, chat_template_kwargs=chat_template_kwargs, eot_ids=eot_ids, max_length=max_length)
        if isinstance(out, str):
            rep.dropped.append({"pair_id": ex.pair_id, "reason": out})
        else:
            rep.kept.append(out)
    return rep


def profile_renderer(profile_name_or_path: str) -> tuple[ModelProfile, Any, set[int]]:
    """Convenience: (profile, verified tokenizer, end-of-turn ids)."""
    profile = load_model_profile(profile_name_or_path)
    tok = load_tokenizer(profile)
    return profile, tok, end_of_turn_ids(tok, profile)
