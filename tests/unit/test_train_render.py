"""Chat-template rendering of preference pairs: serialization, boundary, masking, EOS, drops.

Two tokenizers:
- `byte_tok`: a byte-level fixture tokenizer carrying the Qwen3 chat template text
  (tests/fixtures/train/qwen3_chat_template.jinja, copied from Qwen/Qwen3-0.6B@c1899de,
  Apache-2.0). Needs no model download, so the logic is always tested.
- the real pinned Qwen3 / Gemma 4 tokenizers when they are in the local HF cache
  (tokenizer files only; no model weights are loaded). Skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learning_loop.config import load_model_profile
from learning_loop.records import PreferenceExample
from learning_loop.training.common import load_preference_dataset
from learning_loop.training.render import (
    RenderError,
    end_of_turn_ids,
    load_tokenizer,
    render_completion,
    render_pair,
    render_pairs,
    render_text,
    to_template_messages,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "train"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]
QWEN_KW = {"enable_thinking": False}


def call(cid: str, args: str, name: str = "bash") -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def asst(c: dict, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [c]}


OBS = "[exit code 0]\nSECRET_OBSERVATION_TEXT app.log"
PROMPT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Count ERROR lines."},
    asst(call("call_0", '{"command": "ls"}')),
    {"role": "tool", "tool_call_id": "call_0", "content": OBS},
]


def pair(chosen_args: str, rejected_args: str, pid: str = "p1", prompt: list | None = None) -> PreferenceExample:
    return PreferenceExample(
        pair_id=pid,
        prompt=prompt or PROMPT,
        chosen=[asst(call("call_1", chosen_args))],
        rejected=[asst(call("call_1", rejected_args))],
        tools=TOOLS,
    )


@pytest.fixture(scope="module")
def byte_tok():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<think>", "</think>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = (FIX / "qwen3_chat_template.jinja").read_text()
    return tok


def _cached(profile: str):
    prof = load_model_profile(profile)
    try:
        return prof, load_tokenizer(prof, local_files_only=True)
    except OSError:
        pytest.skip(f"{prof.base_model}@{prof.base_revision[:8]} tokenizer not in the local HF cache")


# qwen3-1.7b shares the 0.6B chat template (same pinned template sha256); both run tokenizer-only.
@pytest.fixture(scope="module", params=["qwen3-0.6b", "qwen3-1.7b"])
def qwen(request):
    return _cached(request.param)


# --------------------------------------------------------------------------- #
# Logic (fixture tokenizer, always runs)
# --------------------------------------------------------------------------- #


def test_arguments_parsed_to_objects_and_ids_kept():
    msgs = to_template_messages([asst(call("abc", '{"command":"ls -la"}'), content=None)])
    tc = msgs[0]["tool_calls"][0]
    assert tc["id"] == "abc" and tc["function"]["arguments"] == {"command": "ls -la"}
    assert msgs[0]["content"] == ""
    for bad in ('{"command": ', '["ls"]', "3"):
        with pytest.raises(RenderError):
            to_template_messages([asst(call("x", bad))])


def test_boundary_masking_eos_and_identical_context(byte_tok):
    eot = end_of_turn_ids(byte_tok)
    p = render_pair(pair('{"command": "grep -c ERROR app.log"}', '{"command": "cat app.log"}'), byte_tok,
                    chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=None)
    assert not isinstance(p, str)
    assert p.prompt_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # identical effective context: one prompt, exact token prefix of both full renders
    for comp in (p.chosen_ids, p.rejected_ids):
        assert comp[-1] in eot  # ends at the end-of-turn token
        assert eot.isdisjoint(comp[:-1])
    full_c = byte_tok(render_text(byte_tok, PROMPT + [asst(call("call_1", '{"command": "grep -c ERROR app.log"}'))], TOOLS,
                                  add_generation_prompt=False, chat_template_kwargs=QWEN_KW), add_special_tokens=False)["input_ids"]
    assert full_c[: len(p.prompt_ids)] == p.prompt_ids
    assert full_c[len(p.prompt_ids): len(p.prompt_ids) + len(p.chosen_ids)] == p.chosen_ids
    # completion-only loss: every prompt position (incl. the tool observation) is masked
    labels = p.labels("chosen")
    assert labels[: len(p.prompt_ids)] == [-100] * len(p.prompt_ids)
    assert labels[len(p.prompt_ids):] == p.chosen_ids
    assert "SECRET_OBSERVATION_TEXT" in p.prompt_text and "SECRET_OBSERVATION_TEXT" not in p.chosen_text
    assert p.chosen_text == '<tool_call>\n{"name": "bash", "arguments": {"command": "grep -c ERROR app.log"}}\n</tool_call><|im_end|>'


def test_trailing_template_tokens_not_trained(byte_tok):
    rc = render_completion(byte_tok, PROMPT, [asst(call("c", '{"command": "ls"}'))], TOOLS,
                           chat_template_kwargs=QWEN_KW, eot_ids=end_of_turn_ids(byte_tok))
    assert byte_tok.decode(rc.trailing_dropped) == "\n"  # template newline after <|im_end|>


def test_argument_spacing_is_canonicalized(byte_tok):
    eot = end_of_turn_ids(byte_tok)
    a = render_completion(byte_tok, PROMPT, [asst(call("c", '{"command":"ls"}'))], TOOLS, chat_template_kwargs=QWEN_KW, eot_ids=eot)
    b = render_completion(byte_tok, PROMPT, [asst(call("c", '{ "command" :  "ls" }'))], TOOLS, chat_template_kwargs=QWEN_KW, eot_ids=eot)
    assert a.completion_ids == b.completion_ids


def test_drops_are_recorded_not_truncated(byte_tok):
    eot = end_of_turn_ids(byte_tok)
    exs = [
        pair('{"command": "grep -c ERROR app.log"}', '{"command": "cat app.log"}', "ok"),
        pair('{"command":"cat app.log"}', '{"command": "cat app.log"}', "same"),  # differs only in spacing
        pair('{"command": ', '{"command": "cat app.log"}', "badjson"),
    ]
    rep = render_pairs(exs, byte_tok, chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=None)
    assert [p.pair_id for p in rep.kept] == ["ok"]
    reasons = {d["pair_id"]: d["reason"] for d in rep.dropped}
    assert reasons["same"].startswith("chosen_equals_rejected")
    assert reasons["badjson"].startswith("invalid_arguments")
    full = rep.kept[0]
    rep2 = render_pairs(exs[:1], byte_tok, chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=full.total_length - 1)
    assert not rep2.kept and rep2.dropped[0]["reason"].startswith("oversize:")
    rep3 = render_pairs(exs[:1], byte_tok, chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=full.total_length)
    assert rep3.kept[0].chosen_ids == full.chosen_ids  # at the limit: kept whole
    summ = rep.summary()
    assert (summ["n_input"], summ["n_kept"], summ["n_dropped"]) == (3, 1, 2)
    assert summ["kept_tokens"] == {"min": full.total_length, "median": full.total_length, "max": full.total_length}
    s2 = rep2.summary()
    assert s2["max_length"] == full.total_length - 1 and s2["kept_tokens"] is None and s2["dropped_by_reason"] == {"oversize": 1}


def test_trainer_records_render_summary_when_everything_is_dropped(tmp_path, monkeypatch):
    from learning_loop.training import run as train_run
    from learning_loop.training.common import NoTrainableExamples

    summary = {"n_kept": 0, "n_dropped": 1, "dropped_by_reason": {"oversize": 1}}

    class AllDropped:
        def train(self, request, work_dir):
            raise NoTrainableExamples("all 1 examples dropped", render=summary)

    monkeypatch.setattr(train_run, "get_trainer", lambda name, **kw: AllDropped())
    from learning_loop.interfaces import TrainRequest
    from learning_loop.training.fixture import base_checkpoint_ref

    req = TrainRequest(run_id="r", cycle=0, dataset_dir=str(tmp_path), incoming=base_checkpoint_ref("qwen3-0.6b"),
                       model_profile="qwen3-0.6b", training_config={"trainer": "trl_dpo"}, seed=1, output_root=str(tmp_path / "out"), device="cpu")
    (tmp_path / "req.json").write_text(req.model_dump_json())
    assert train_run.main(["--request", str(tmp_path / "req.json"), "--work-dir", str(tmp_path / "w")]) == 3
    res = json.loads((tmp_path / "w" / "result.json").read_text())
    assert res["status"] == "no_trainable_examples" and res["render"] == summary


def test_multi_call_or_non_assistant_completion_rejected(byte_tok):
    eot = end_of_turn_ids(byte_tok)
    ex = pair('{"command": "a"}', '{"command": "b"}')
    two = asst(call("x", '{"command": "a"}'))
    two["tool_calls"].append(call("y", '{"command": "b"}'))
    bad = ex.model_copy(update={"chosen": [two]})
    assert render_pair(bad, byte_tok, chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=None).startswith("invalid_chosen")
    bad2 = ex.model_copy(update={"prompt": PROMPT[:2] + [PROMPT[2]]})
    assert render_pair(bad2, byte_tok, chat_template_kwargs=QWEN_KW, eot_ids=eot, max_length=None).startswith("invalid_prompt")


def test_template_that_breaks_prefix_fails_loudly(byte_tok):
    import copy

    tok = copy.deepcopy(byte_tok)
    # a template that renders the system prompt differently when a generation prompt is requested
    tok.chat_template = "{% for m in messages %}{{ m.role }}{% if add_generation_prompt %}!{% endif %}|{% endfor %}{% if add_generation_prompt %}A:{% endif %}"
    with pytest.raises(RenderError):
        render_completion(tok, PROMPT, [asst(call("c", '{"command": "ls"}'))], TOOLS, chat_template_kwargs={},
                          eot_ids=end_of_turn_ids(tok))


# --------------------------------------------------------------------------- #
# Real pinned tokenizers (cache only)
# --------------------------------------------------------------------------- #


def test_qwen3_exact_serialization_and_ids(qwen):
    prof, tok = qwen
    eot = end_of_turn_ids(tok, prof)
    assert eot == {151645, 151643}  # <|im_end|>, <|endoftext|> (generation_config)
    ex = pair('{"command":"grep -c ERROR /app/logs/app.log"}', '{"command":"cat /app/logs/app.log"}')
    p = render_pair(ex, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=eot, max_length=4096)
    assert not isinstance(p, str)
    assert p.chosen_text == '<tool_call>\n{"name": "bash", "arguments": {"command": "grep -c ERROR /app/logs/app.log"}}\n</tool_call><|im_end|>'
    assert p.rejected_text == '<tool_call>\n{"name": "bash", "arguments": {"command": "cat /app/logs/app.log"}}\n</tool_call><|im_end|>'
    assert p.chosen_ids[-1] == tok.convert_tokens_to_ids("<|im_end|>")
    assert p.prompt_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # the previous tool call is rendered as an object with the same canonical spacing
    assert '{"name": "bash", "arguments": {"command": "ls"}}' in p.prompt_text
    # Qwen3's template does not render tool-call ids: renaming them changes nothing
    renamed = json.loads(json.dumps(ex.model_dump()).replace("call_0", "zz_other").replace("call_1", "zz_new"))
    p2 = render_pair(PreferenceExample.model_validate(renamed), tok, chat_template_kwargs=prof.chat_template_kwargs,
                     eot_ids=eot, max_length=4096)
    assert (p2.prompt_ids, p2.chosen_ids, p2.rejected_ids) == (p.prompt_ids, p.chosen_ids, p.rejected_ids)


def test_qwen3_fixture_dataset_renders(qwen):
    prof, tok = qwen
    exs, sha, info = load_preference_dataset(FIX / "fixture_prefs")
    assert info["provenance_kinds"] == {"fixture": 3}
    rep = render_pairs(exs, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=end_of_turn_ids(tok, prof), max_length=1024)
    assert len(rep.kept) == 3 and not rep.dropped
    for p in rep.kept:
        assert p.chosen_ids != p.rejected_ids


GEMMA = ["gemma-4-e4b-it", "gemma-4-e2b-it"]  # different pinned templates, same tool-call syntax


@pytest.mark.parametrize("profile", GEMMA)
def test_gemma4_template_boundary_tokenizer_only(profile):
    prof, tok = _cached(profile)
    eot = end_of_turn_ids(tok, prof)
    assert {1, 106, 50} <= eot
    ex = pair('{"command":"grep -c ERROR app.log","b":1}', '{"command":"cat app.log"}')
    p = render_pair(ex, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=eot, max_length=4096)
    assert not isinstance(p, str)
    assert p.chosen_text == '<|tool_call>call:bash{b:1,command:<|"|>grep -c ERROR app.log<|"|>}<tool_call|><|tool_response>'
    assert p.chosen_ids[-1] == tok.convert_tokens_to_ids("<|tool_response>")


@pytest.mark.parametrize("profile", GEMMA)
def test_gemma4_parser_roundtrips_the_pinned_template(profile):
    """What the template renders for a tool call (= what the model is trained to emit) parses back
    to the same arguments, and the parsed call re-renders to the same tokens."""
    from learning_loop.serving.tool_parse import parse_gemma4

    prof, tok = _cached(profile)
    eot = end_of_turn_ids(tok, prof)
    args = '{"command":"grep -c \\"ERROR\\" app.log | sort, {x}","flags":["-a",2],"n":3,"o":{"t":true,"f":false,"z":1.5}}'
    ex = pair(args, '{"command":"cat app.log"}')
    p = render_pair(ex, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=eot, max_length=4096)
    text = tok.decode(p.chosen_ids[:-1], skip_special_tokens=False)  # the server strips the stop token
    parsed = parse_gemma4(text, "s")
    assert parsed.parse_errors == [] and parsed.content == ""
    (tc,) = parsed.tool_calls
    assert json.loads(tc["function"]["arguments"]) == json.loads(args)
    again = pair(tc["function"]["arguments"], '{"command":"cat app.log"}')
    q = render_pair(again, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=eot, max_length=4096)
    assert q.chosen_ids == p.chosen_ids


def test_template_pin_is_enforced(qwen):
    prof, _ = qwen
    bad = prof.model_copy(update={"chat_template_sha256": "0" * 64})
    with pytest.raises(RenderError):
        load_tokenizer(bad, local_files_only=True)


def test_intervention_tokens_match_training_render(qwen):
    from learning_loop.token_count import fixture_intervention_tokens, get_counter, intervention_tokens

    prof, tok = qwen
    ex = pair('{"command": "grep -c ERROR app.log"}', '{"command": "cat app.log"}')
    p = render_pair(ex, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=end_of_turn_ids(tok, prof), max_length=None)
    n, src = intervention_tokens(prof, ex.prompt, ex.chosen[0], ex.tools)
    assert n == len(p.chosen_ids)
    assert src.startswith(f"hf_template:{prof.name}:tokenizer=") and "template=a55ee1b16601" in src
    assert get_counter(prof.name)(ex.prompt, ex.chosen[0], ex.tools) == (n, src)
    fn, fsrc = fixture_intervention_tokens(ex.prompt, ex.chosen[0], ex.tools)
    assert fsrc.startswith("fixture_estimate") and fn > 0
    assert get_counter(None) is fixture_intervention_tokens
