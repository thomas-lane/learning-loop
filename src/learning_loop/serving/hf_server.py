"""Minimal OpenAI-compatible server for one immutable checkpoint (hf_transformers backend).

    python -m learning_loop.serving.hf_server --profile qwen3-0.6b \
        [--checkpoint-dir runs/<run>/checkpoints/<id>] --port 8765

- Loads base_model@base_revision and, if given, the PEFT adapter of a published
  checkpoint (its adapter sha256 is re-checked). The served model name is the
  checkpoint id (`base` without an adapter); requests for any other `model` get
  404, so a run can never talk to a mutable "latest".
- Prompts are rendered with training/render.py (the profile's pinned template and
  `chat_template_kwargs`), i.e. exactly the rendering used for DPO training.
- Native tool calls are parsed into OpenAI `tool_calls` (JSON-string arguments);
  reasoning goes to `reasoning_content` only when the model produced it.
- Sampling follows OpenAI semantics with an explicit GenerationConfig (model
  generation defaults such as top_k are NOT applied): temperature default 1.0,
  0 = greedy; top_p default 1.0. `seed` seeds torch before generation.
- Usage is exact from the tokenizer: prompt tokens as rendered; completion tokens
  as generated, including the end-of-turn token when one was produced.
- Serial: one request at a time (single-threaded HTTPServer).
- GET /health, GET /v1/models, POST /v1/chat/completions. Extra diagnostics are in
  the response's `learning_loop` field (raw completion text, parse errors, sampling).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from ..core.config import ModelProfile, load_model_profile
from ..core.records import CheckpointRecord
from ..core.storage import read_json
from .tool_parse import PARSERS, parse_completion


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


class Engine(Protocol):
    served_model: str

    def info(self) -> dict[str, Any]: ...

    def complete(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


def validate_request(body: dict[str, Any], served_model: str) -> dict[str, Any]:
    """Normalize an OpenAI chat request; raise ApiError for anything unsupported."""
    if not isinstance(body, dict):
        raise ApiError(400, "request body must be a JSON object")
    model = body.get("model")
    if model is not None and model != served_model:
        raise ApiError(404, f"model {model!r} is not served here (serving {served_model!r})", "model_not_found")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ApiError(400, "messages must be a non-empty list")
    if body.get("stream"):
        raise ApiError(400, "stream=true is not supported")
    if body.get("n") not in (None, 1):
        raise ApiError(400, "n > 1 is not supported")
    if body.get("tool_choice") not in (None, "auto"):
        raise ApiError(400, "only tool_choice=auto is supported")
    temperature = body.get("temperature")
    temperature = 1.0 if temperature is None else float(temperature)
    top_p = body.get("top_p")
    top_p = 1.0 if top_p is None else float(top_p)
    if temperature < 0 or not (0 < top_p <= 1):
        raise ApiError(400, "invalid temperature/top_p")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    seed = body.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ApiError(400, "seed must be an integer")
    return {
        "messages": msgs,
        "tools": body.get("tools") or None,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "seed": seed,
    }


class ServingCore:
    """What every engine shares: the profile's pinned tokenizer and template, the served checkpoint
    identity (a published adapter re-checked against its recorded hash, or the base model), and
    request handling around generation (rendering with training/render.py, context check, our
    tool-call parser, usage counted with the training tokenizer). Engines only generate token ids."""

    engine_name = "?"

    def __init__(self, profile: ModelProfile, checkpoint_dir: Path | None, base_checkpoint_id: str,
                 zero_lora: dict[str, Any] | None, default_max_tokens: int):
        from ..training.modeling import verify_adapter_dir
        from ..training.render import chat_template_sha256, end_of_turn_ids, load_tokenizer

        if profile.tool_call_format not in PARSERS:
            raise NotImplementedError(f"hf_server has no tested parser for tool_call_format={profile.tool_call_format!r}")
        if checkpoint_dir is not None and zero_lora:
            raise ValueError("--zero-lora applies to base checkpoints only")
        self.profile = profile
        self.tok = load_tokenizer(profile)
        self.template_sha = chat_template_sha256(self.tok)
        self.eot = end_of_turn_ids(self.tok, profile)
        self.zero_lora = zero_lora
        self.default_max_tokens = default_max_tokens
        # One tokenizer object is shared by concurrent requests (vLLM engine); fast tokenizers are
        # not safe to use from several threads at once, so rendering/encoding/decoding are serialized.
        self._tok_lock = threading.Lock()
        self.record: CheckpointRecord | None = None
        self.adapter_dir: Path | None = None
        if checkpoint_dir is not None:
            rec = CheckpointRecord.model_validate(read_json(Path(checkpoint_dir) / "checkpoint.json"))
            ref = rec.checkpoint
            if (ref.base_model, ref.base_revision) != (profile.base_model, profile.base_revision):
                raise ValueError(f"checkpoint base {ref.base_model}@{ref.base_revision} does not match profile {profile.name}")
            if ref.adapter_path is not None and Path(ref.adapter_path).resolve() != Path(checkpoint_dir).resolve():
                ref = ref.model_copy(update={"adapter_path": str(Path(checkpoint_dir).resolve())})  # copied/moved dir
            self.adapter_dir = verify_adapter_dir(ref)
            self.record = rec
            self.served_model = ref.checkpoint_id
            self.adapter_sha256 = ref.adapter_sha256
        else:
            self.served_model = base_checkpoint_id
            self.adapter_sha256 = None

    def info(self) -> dict[str, Any]:
        return {
            "model": self.served_model,
            "engine": self.engine_name,
            "profile": self.profile.name,
            "base_model": self.profile.base_model,
            "base_revision": self.profile.base_revision,
            "adapter_sha256": self.adapter_sha256,
            "zero_lora": self.zero_lora,
            "chat_template_sha256": self.template_sha,
            "chat_template_kwargs": self.profile.chat_template_kwargs,
            "max_context": self.max_context,
            **self.engine_info(),
        }

    def engine_info(self) -> dict[str, Any]:
        return {}

    def generate(self, ids: list[int], req: dict[str, Any], max_new: int) -> tuple[list[int], dict[str, Any]]:
        """Generated token ids (ending with an end-of-turn id when one was produced) and the
        sampling settings actually applied."""
        raise NotImplementedError

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        from ..training.render import RenderError, encode, render_text

        req = validate_request(body, self.served_model)
        try:
            with self._tok_lock:
                prompt_text = render_text(
                    self.tok, req["messages"], req["tools"], add_generation_prompt=True,
                    chat_template_kwargs=self.profile.chat_template_kwargs,
                )
                ids = encode(self.tok, prompt_text)
        except RenderError as e:
            raise ApiError(400, f"cannot render messages: {e}") from e
        max_new = req["max_tokens"] or self.default_max_tokens
        if len(ids) + max_new > self.max_context:
            raise ApiError(
                400,
                f"the request exceeds the available context size ({len(ids)} prompt + {max_new} max_tokens > {self.max_context})",
                "context_length_exceeded",
            )
        t0 = time.time()
        new, sampling = self.generate(ids, req, max_new)
        gen_sec = time.time() - t0
        stopped = bool(new) and new[-1] in self.eot
        text_ids = new[:-1] if stopped else new
        with self._tok_lock:
            text = self.tok.decode(text_ids, skip_special_tokens=False)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        parsed = parse_completion(self.profile.tool_call_format, text, rid)
        if not stopped and len(new) >= max_new:
            finish = "length"
        elif parsed.tool_calls:
            finish = "tool_calls"
        else:
            finish = "stop"
        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model,
            "system_fingerprint": f"{self.engine_name}:{self.profile.name}:{self.served_model}",
            "choices": [{"index": 0, "message": parsed.message(), "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(new), "total_tokens": len(ids) + len(new)},
            "learning_loop": {
                "raw_completion_text": text,
                "parse_errors": parsed.parse_errors,
                "seed": req["seed"],
                "seed_applied": req["seed"] is not None,
                "sampling": sampling,
                "generation_sec": round(gen_sec, 4),
                "engine": self.engine_name,
                "adapter_sha256": self.adapter_sha256,
            },
        }

    def close(self) -> None:
        pass


class HFEngine(ServingCore):
    """base@revision (+ PEFT adapter, or an all-zero LoRA) on one device; not merged, eval mode."""

    engine_name = "hf_transformers"

    def __init__(
        self,
        profile: ModelProfile,
        checkpoint_dir: Path | None,
        device: str = "auto",
        allow_cpu_fallback: bool = False,
        dtype: str | None = None,
        base_checkpoint_id: str = "base",
        zero_lora: dict[str, Any] | None = None,
        default_max_tokens: int = 1024,
        max_context: int | None = None,
    ):
        import torch  # noqa: F401  (train extra)

        from ..training.common import resolve_device
        from ..training.modeling import apply_zero_lora, load_adapter, load_base_model

        super().__init__(profile, checkpoint_dir, base_checkpoint_id, zero_lora, default_max_tokens)
        self.dev = resolve_device(device, allow_cpu_fallback)
        self.device = self.dev["device"]
        self.dtype = dtype or profile.training_dtype
        base = load_base_model(profile, self.device, self.dtype)
        if self.adapter_dir is not None:
            self.model = load_adapter(base, self.adapter_dir, trainable=False)
        else:
            self.model = apply_zero_lora(base, zero_lora) if zero_lora else base
        self.model.eval()
        cfg_ctx = getattr(self.model.config, "max_position_embeddings", None)
        self.max_context = max_context or cfg_ctx or 32768
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else next(iter(self.eot))

    def engine_info(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype, "concurrency": 1}

    def generate(self, ids: list[int], req: dict[str, Any], max_new: int) -> tuple[list[int], dict[str, Any]]:
        import torch
        from transformers import GenerationConfig

        greedy = req["temperature"] == 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": not greedy,
            "eos_token_id": sorted(self.eot),
            "pad_token_id": self.pad_id,
        }
        if not greedy:
            gen_kwargs.update(temperature=req["temperature"], top_p=req["top_p"], top_k=0)
        gcfg = GenerationConfig(**gen_kwargs)
        if req["seed"] is not None:
            torch.manual_seed(req["seed"])
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=gcfg)
        return out[0, len(ids):].tolist(), {k: v for k, v in gen_kwargs.items() if k not in ("eos_token_id", "pad_token_id")}

    def close(self) -> None:
        from ..training.modeling import free_memory

        self.model = None  # type: ignore[assignment]
        free_memory()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class _Handler(BaseHTTPRequestHandler):
    server_version = "learning-loop-hf/1"
    engine: Engine  # set on the server class

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, code: str) -> None:
        self._send(status, {"error": {"message": message, "type": code, "code": code}})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("hf_server: " + (fmt % args) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        eng = self.server.engine  # type: ignore[attr-defined]
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._send(200, {"status": "ok", **eng.info()})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": eng.served_model, "object": "model", "owned_by": "learning_loop", "meta": eng.info()}]})
        else:
            self._error(404, f"no route {path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        eng = self.server.engine  # type: ignore[attr-defined]
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._error(404, f"no route {path}", "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._error(400, f"invalid JSON body: {e}", "invalid_request_error")
            return
        try:
            self._send(200, eng.complete(body))
        except ApiError as e:
            self._error(e.status, e.message, e.code)
        except Exception as e:  # noqa: BLE001 - reported to the client as a 500
            self._error(500, f"{type(e).__name__}: {e}", "server_error")


def make_server(engine: Engine, host: str, port: int, threaded: bool = False) -> HTTPServer:
    srv = (ThreadingHTTPServer if threaded else HTTPServer)((host, port), _Handler)
    srv.engine = engine  # type: ignore[attr-defined]
    return srv


def serve(engine: Engine, host: str, port: int, threaded: bool = False) -> None:
    srv = make_server(engine, host, port, threaded=threaded)
    stop = threading.Event()

    def _stop(signum: int, _frame: Any) -> None:
        if not stop.is_set():
            stop.set()
            threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(json.dumps({"event": "ready", "host": host, "port": srv.server_address[1], **engine.info()}), flush=True)
    try:
        srv.serve_forever(poll_interval=0.2)
    finally:
        srv.server_close()
        engine.close()
        print(json.dumps({"event": "stopped"}), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learning_loop.serving.hf_server")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None, help="published checkpoint dir; omit to serve the base model")
    ap.add_argument("--base-checkpoint-id", default="base")
    ap.add_argument("--zero-lora", default=None, help="JSON {r, alpha, target_modules, exclude_modules}: serve the base model "
                    "through an all-zero LoRA of this shape (same outputs, same adapter overhead as trained checkpoints)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--default-max-tokens", type=int, default=1024)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--engine", default="hf_transformers", choices=["hf_transformers", "vllm"],
                    help="who generates: this process (transformers) or a vLLM child process fed token ids")
    ap.add_argument("--engine-package", default=None, help="vllm: pip requirement of the engine environment, e.g. vllm==0.30.0")
    ap.add_argument("--concurrency", type=int, default=1, help="vllm: concurrent requests (vLLM batches them); 1 = unbatched")
    ap.add_argument("--engine-arg", action="append", default=[], help="vllm: extra vLLM server flag (repeatable)")
    ap.add_argument("--engine-start-timeout", type=float, default=1800.0, help="vllm: seconds to install/start the engine")
    a = ap.parse_args(argv)
    profile = load_model_profile(a.profile)
    zero = json.loads(a.zero_lora) if a.zero_lora else None
    if a.engine == "vllm":
        from .vllm_engine import VllmEngine

        if not a.engine_package:
            ap.error("--engine vllm needs --engine-package")
        engine: ServingCore = VllmEngine(
            profile, a.checkpoint_dir, a.engine_package, base_checkpoint_id=a.base_checkpoint_id, zero_lora=zero,
            default_max_tokens=a.default_max_tokens, max_context=a.max_context, concurrency=a.concurrency,
            engine_args=a.engine_arg, start_timeout_sec=a.engine_start_timeout,
        )
    else:
        engine = HFEngine(
            profile, a.checkpoint_dir, device=a.device, allow_cpu_fallback=a.allow_cpu_fallback,
            dtype=a.dtype, base_checkpoint_id=a.base_checkpoint_id, default_max_tokens=a.default_max_tokens,
            zero_lora=zero, max_context=a.max_context,
        )
    serve(engine, a.host, a.port, threaded=a.concurrency > 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
