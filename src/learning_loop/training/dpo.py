"""`trainer: trl_dpo` — LoRA DPO with TRL 1.14.1 + PEFT, continuing the incoming adapter.

What was verified in the pinned TRL source (trl/trainer/dpo_trainer.py) and why
this module does not rely on its defaults:

- `DPOTrainer(peft_config=...)` wraps a plain model in a NEW adapter; its
  reference is then the adapter-disabled base. That is only the incoming learner
  in cycle 0. Passing an already-loaded `PeftModel` makes TRL copy the adapter
  into a frozen "ref" adapter (PEFT >= 0.20), which would be correct, but the
  choice is implicit and version-dependent.
- Here the reference is explicit: reference log-probs are precomputed with the
  incoming learner INCLUDING its adapter (eval mode, no grad), using TRL's own
  collator and masked log-softmax, and cached under a key of (reference
  checkpoint id + adapter sha, example sha, token-ids sha, tokenizer sha,
  template sha, template kwargs, dtype, log-prob path). A subclass feeds them to TRL through the
  `precompute_ref_log_probs` path (`ref_chosen_logps` / `ref_rejected_logps`
  columns), so TRL never computes a reference itself. The unused "ref" adapter
  copy TRL adds is deleted before training.
- Tokenization is ours (training/render.py): TRL's tokenize_fn only warns on a
  prompt/completion prefix mismatch and keeps trailing template tokens; ours
  fails on mismatch, cuts the completion at the end-of-turn token and drops
  oversize pairs with a recorded reason (TRL would truncate).
- Sanity check of the resolved reference: before the first update the policy
  equals the reference, so TRL's logged first-step `rewards/*` must be ~0. A
  wrong (e.g. adapter-disabled) reference in a later cycle fails this check.

- One numerical path for every log-prob: Trainer mixed precision is never enabled
  (bf16=fp16=False; weights already carry the profile dtype), so accelerate does not
  patch the model forward with autocast + convert_outputs_to_fp32
  (accelerate/accelerator.py `prepare_model`, only when `native_amp`), and TRL's loss
  forward is the plain forward our reference, trained and reload log-probs use. A forward
  hook upcasts lm_head output to float32 (modeling.ensure_fp32_logits), so TRL's
  selective log-softmax and ours both run on fp32 logits and sum in fp32. The trained
  log-probs are taken on `accelerator.unwrap_model(..., keep_fp32_wrapper=False)`.
  CPU, MPS and CUDA (RTX 3090, Qwen3-0.6B) are exercised.
- Only the completion window goes through lm_head (`logits_to_keep`, see
  `_compute_loss` and modeling.completion_window), on every log-prob path: the loss
  equals TRL's full-logits loss (unit-tested) without float32 logits for every prompt
  token, which is what limits long prompts with a 262k-token vocabulary.

Optimizer/scheduler are fresh every cycle (new Trainer); adapter weights continue.
Interrupted stages resume from the stage's own Trainer checkpoint (adapter,
optimizer, scheduler, python/numpy/torch-CPU(/CUDA) RNG as saved by
transformers.Trainer; MPS RNG is not saved by transformers and is unused
because dropout is disabled). On CUDA a resumed stage matches an uninterrupted one only up to
run-to-run noise: the efficient attention kernels are nondeterministic, and a LoRA's first Adam
steps amplify that noise (the training tests force deterministic kernels to check resume exactly).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from ..interfaces import TrainRequest
from ..records import CheckpointRecord, CheckpointRef
from ..storage import JsonlAppender, atomic_write_json, now_iso, read_json, read_jsonl, sha256_json
from .common import (
    NoTrainableExamples,
    TrainingRequestError,
    adapter_sha256,
    derive_checkpoint_id,
    discard_staging,
    existing_publication,
    load_preference_dataset,
    make_staging_dir,
    publish,
    request_sha256,
    resolve_device,
    resolve_profile,
    resolve_training_config,
)
from .fixture import SimulatedInterruption
from .modeling import (
    ADAPTER_FILES,
    adapter_state_cpu,
    ensure_fp32_logits,
    free_memory,
    load_adapter,
    load_adapter_file_state,
    load_base_model,
    completion_window,
    max_abs_diff,
    sequence_logps,
    verify_adapter_dir,
)
from .render import RenderedPair, chat_template_sha256, end_of_turn_ids, load_tokenizer, render_pairs, tokenizer_sha256

# Declared tolerances (nats for summed sequence log-probs).
LOAD_CHECK_LOGP_ATOL = 1e-3  # published adapter reloaded vs in-memory trained model
LOAD_CHECK_PARAM_ATOL = 0.0  # adapter tensors must round-trip exactly
REFERENCE_IDENTITY_ATOL = 2e-2  # |policy - reference| log-ratio at the first step (before any update)
N_CHECK_EXAMPLES = 2
# Identifies how log-probs are computed; part of the reference cache key so entries made by an
# older numerical path (e.g. bf16 log-softmax) are never reused.
LOGP_PATH = "completion_window_fp32_logits_selective_log_softmax_fp32_sum_v2"


class ReferenceMismatch(RuntimeError):
    pass


class LoadCheckFailed(RuntimeError):
    pass


def reference_cache_key(
    ref: CheckpointRef, pair: RenderedPair, tok_sha: str, tmpl_sha: str, template_kwargs: dict[str, Any], dtype: str
) -> dict[str, Any]:
    return {
        "reference_checkpoint_id": ref.checkpoint_id,
        "reference_adapter_sha256": ref.adapter_sha256,
        "base": f"{ref.base_model}@{ref.base_revision}",
        "example_sha256": pair.example_sha256,
        "token_ids_sha256": pair.token_ids_sha256,
        "tokenizer_sha256": tok_sha,
        "chat_template_sha256": tmpl_sha,
        "chat_template_kwargs": template_kwargs,
        "dtype": dtype,
        "logp_path": LOGP_PATH,
    }


class ReferenceCache:
    """One JSON file per key under `root`: {"key": ..., "chosen": float, "rejected": float}."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def _path(self, key: dict[str, Any]) -> Path:
        return self.root / f"{sha256_json(key)}.json"

    def get(self, key: dict[str, Any]) -> tuple[float, float] | None:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        d = read_json(p)
        if d.get("key") != key:
            self.misses += 1
            return None
        self.hits += 1
        return float(d["chosen"]), float(d["rejected"])

    def put(self, key: dict[str, Any], chosen: float, rejected: float, meta: dict[str, Any]) -> None:
        atomic_write_json(self._path(key), {"key": key, "chosen": chosen, "rejected": rejected, "meta": meta})


def _make_trainer_class():
    from trl import DPOTrainer

    class PrecomputedRefDPOTrainer(DPOTrainer):
        """DPOTrainer fed with our tokenized rows and our reference log-probs."""

        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            need = {"prompt_ids", "chosen_ids", "rejected_ids", "ref_chosen_logps", "ref_rejected_logps"}
            missing = need - set(dataset.column_names)
            if missing:
                raise ValueError(f"pre-tokenized dataset is missing columns {sorted(missing)}")
            return dataset

        def _precompute_ref_logps(self, dataset, name, batch_size):
            return dataset  # already attached by TrlDpoTrainer (incoming learner incl. adapter)

        def _compute_loss(self, model, inputs, return_outputs):
            # TRL's loss on the completion window only (modeling.completion_window): the model
            # sees the full sequence but lm_head runs on the last k positions, and TRL's labels
            # and completion mask are cropped to match. attention_mask stays full (TRL counts
            # tokens with it). sequence_logps uses the same window for every other log-prob.
            k = completion_window(inputs["completion_mask"])
            full_ids = inputs["input_ids"]
            cropped = {**inputs, "input_ids": full_ids[:, -k:], "completion_mask": inputs["completion_mask"][:, -k:]}

            def forward(**kw):
                return model(**{**kw, "input_ids": full_ids, "logits_to_keep": k})

            return super()._compute_loss(forward, cropped, return_outputs)

    return PrecomputedRefDPOTrainer


def _callbacks(log_path: Path, interrupt_after: int | None, resumed: bool):
    from transformers import TrainerCallback

    out = JsonlAppender(log_path)  # survives interruption; a resumed stage keeps earlier steps' logs

    class Recorder(TrainerCallback):
        def on_log(self, args, state, control, logs_=None, **kw):
            payload = kw.get("logs", logs_) or {}
            out.append({"step": state.global_step, **{k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}})

    class Interrupter(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            if interrupt_after is not None and not resumed and state.global_step == interrupt_after:
                raise SimulatedInterruption(f"training interrupted after optimizer step {state.global_step}")

    return [Recorder(), Interrupter()]


class TrlDpoTrainer:
    name = "trl_dpo"

    def __init__(
        self,
        ref_cache_dir: str | Path | None = None,
        save_every: int | None = None,
        record_base_logps: bool = True,
        _test_interrupt_after_step: int | None = None,
    ):
        self.ref_cache_dir = Path(ref_cache_dir) if ref_cache_dir else None
        self.save_every = save_every
        self.record_base_logps = record_base_logps
        self._interrupt_after = _test_interrupt_after_step

    # ------------------------------------------------------------------ #

    def train(self, request: TrainRequest, work_dir: Path) -> CheckpointRecord:
        import torch  # noqa: F401  (fail early without the train extra)
        from transformers import set_seed
        from transformers.trainer_utils import get_last_checkpoint

        t_start = time.time()
        phases: dict[str, float] = {}  # wall seconds of the non-GPU-bound steps, for diagnosing stage time
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg = resolve_training_config(request.training_config, self.name)
        inc = request.incoming
        profile = resolve_profile(request.model_profile, inc.model_profile, inc.base_model, inc.base_revision)
        examples, ds_sha, ds_info = load_preference_dataset(request.dataset_dir)
        if not examples:
            raise NoTrainableExamples(f"{request.dataset_dir}: no preference examples")
        tc = cfg.model_dump()
        cid = derive_checkpoint_id(request.run_id, request.cycle, ds_sha, inc.checkpoint_id, tc, request.seed, self.name)
        done = existing_publication(Path(request.output_root), cid, ds_sha)
        if done is not None:
            return done

        # fail fast (before any tokenizer/model load) on an unavailable or unsupported device
        dev = resolve_device(request.device, request.allow_cpu_fallback, profile.supported_train_devices)
        device = dev["device"]

        # ---- stage identity / resume ------------------------------------------------
        req_sha = request_sha256(request.model_dump())
        req_path = work_dir / "request.json"
        if req_path.exists():
            prev = read_json(req_path)
            if prev.get("request_sha256") != req_sha:
                raise TrainingRequestError(f"{work_dir} belongs to a different training request; use a fresh work dir")
        else:
            atomic_write_json(req_path, {"request_sha256": req_sha, "request": request.model_dump(), "checkpoint_id": cid})
        trainer_dir = work_dir / "trainer"
        resume_ckpt = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None

        # ---- render -----------------------------------------------------------------
        t0 = time.time()
        tok = load_tokenizer(profile)
        tok_sha, tmpl_sha = tokenizer_sha256(tok), chat_template_sha256(tok)
        report = render_pairs(
            examples, tok, chat_template_kwargs=profile.chat_template_kwargs,
            eot_ids=end_of_turn_ids(tok, profile), max_length=cfg.dpo.max_length,
        )
        atomic_write_json(work_dir / "render_report.json", report.summary())
        phases["tokenizer_and_render"] = time.time() - t0
        if not report.kept:
            raise NoTrainableExamples(f"all {len(examples)} examples dropped: {report.summary()['dropped_by_reason']}", render=report.summary())
        pairs = report.kept

        # ---- model: continue the incoming adapter, or a new LoRA on the base ----------
        set_seed(request.seed)
        t0 = time.time()
        base = load_base_model(profile, device)
        phases["load_model"] = time.time() - t0
        t0 = time.time()
        lora_targets = cfg.lora.target_modules or profile.lora_target_modules
        if not lora_targets:
            raise TrainingRequestError("no LoRA target modules (training config or model profile)")
        init: dict[str, Any] = {}
        if inc.adapter_path is not None:
            adir = verify_adapter_dir(inc)
            inc_cfg = json.loads((adir / "adapter_config.json").read_text())
            want = {"r": cfg.lora.r, "lora_alpha": cfg.lora.alpha, "target_modules": sorted(lora_targets),
                    "exclude_modules": profile.lora_exclude_modules}
            got = {"r": inc_cfg.get("r"), "lora_alpha": inc_cfg.get("lora_alpha"), "target_modules": sorted(inc_cfg.get("target_modules") or []),
                   "exclude_modules": inc_cfg.get("exclude_modules")}
            if want != got:
                raise TrainingRequestError(f"incoming adapter LoRA config {got} != training config {want}")
            model = load_adapter(base, adir, trainable=True)
            diff = max_abs_diff(adapter_state_cpu(model), load_adapter_file_state(adir))
            init = {"adapter_init": "continue_incoming", "initial_adapter_max_abs_diff_vs_incoming": diff,
                    "initial_adapter_matches_incoming": diff == 0.0}
            if diff != 0.0:
                raise TrainingRequestError(f"loaded adapter differs from incoming files (max abs diff {diff})")
        else:
            from peft import LoraConfig, get_peft_model

            lcfg = LoraConfig(
                r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                target_modules=list(lora_targets), exclude_modules=profile.lora_exclude_modules, task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, lcfg)
            init = {"adapter_init": "new_lora_on_base", "initial_adapter_matches_incoming": None}
        ensure_fp32_logits(model)  # before the reference, TRL's loss forward and the trained log-probs
        phases["adapter_setup"] = time.time() - t0

        # ---- reference log-probs: incoming learner incl. adapter, cached ------------------
        cache = ReferenceCache(self.ref_cache_dir or (work_dir / "ref_logp_cache"))
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ref_vals: list[tuple[float, float]] = []
        to_compute: list[int] = []
        keys = [reference_cache_key(inc, p, tok_sha, tmpl_sha, profile.chat_template_kwargs, profile.training_dtype) for p in pairs]
        for i, k in enumerate(keys):
            hit = cache.get(k)
            ref_vals.append(hit if hit is not None else (math.nan, math.nan))
            if hit is None:
                to_compute.append(i)
        t_ref = time.time()
        if to_compute:
            computed = sequence_logps(model, [pairs[i] for i in to_compute], device, pad_id)
            for i, (c, r) in zip(to_compute, computed):
                ref_vals[i] = (c, r)
                cache.put(keys[i], c, r, {"pair_id": pairs[i].pair_id, "device": device, "created_at": now_iso()})
        ref_sec = time.time() - t_ref
        check_pairs = pairs[:N_CHECK_EXAMPLES]
        ref_check: dict[str, Any] = {
            "reference_checkpoint_id": inc.checkpoint_id,
            "reference_includes_adapter": inc.adapter_path is not None,
            "reference_logps": [list(v) for v in ref_vals[: len(check_pairs)]],
        }
        if self.record_base_logps:
            with model.disable_adapter():
                base_lp = sequence_logps(model, check_pairs, device, pad_id)
            ref_check["base_model_logps"] = [list(v) for v in base_lp]
            ref_check["reference_minus_base_max_abs"] = max(
                abs(a - b) for rv, bv in zip(ref_vals, base_lp) for a, b in zip(rv, bv)
            )

        # ---- TRL DPO -------------------------------------------------------------------
        from datasets import Dataset
        from trl import DPOConfig

        rows = [{**p.row(), "ref_chosen_logps": rv[0], "ref_rejected_logps": rv[1]} for p, rv in zip(pairs, ref_vals)]
        ds = Dataset.from_list(rows)
        steps = cfg.optimizer_steps
        save_every = self.save_every or (1 if steps <= 20 else math.ceil(steps / 10))
        args = DPOConfig(
            output_dir=str(trainer_dir),
            max_steps=steps,
            per_device_train_batch_size=cfg.dpo.per_device_batch_size,
            gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
            learning_rate=cfg.dpo.learning_rate,
            lr_scheduler_type="linear",
            warmup_steps=cfg.dpo.warmup_steps,
            max_grad_norm=cfg.dpo.max_grad_norm,
            optim="adamw_torch",
            beta=cfg.dpo.beta,
            loss_type=[cfg.dpo.loss_type],
            max_length=None,  # oversize pairs were dropped, never truncated
            precompute_ref_log_probs=True,
            disable_dropout=True,
            seed=request.seed,
            data_seed=request.seed,
            logging_steps=1,
            save_strategy="steps",
            save_steps=save_every,
            save_total_limit=1,
            save_only_model=False,
            report_to="none",
            disable_tqdm=True,
            use_cpu=(device == "cpu"),
            dataloader_pin_memory=False,
            # No mixed precision: the weights already have the profile dtype and logits are upcast
            # by the fp32 hook, so TRL's forward is the same plain forward sequence_logps uses.
            bf16=False,
            fp16=False,
            # Recomputes activations in the backward pass (same values; dropout is disabled).
            gradient_checkpointing=cfg.dpo.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        log_path = work_dir / "train_logs.jsonl"
        t0 = time.time()
        Trainer = _make_trainer_class()
        trainer = Trainer(
            model=model, args=args, train_dataset=ds, processing_class=tok,
            callbacks=_callbacks(log_path, self._interrupt_after, resumed=resume_ckpt is not None),
        )
        if "ref" in getattr(trainer.model, "peft_config", {}):
            trainer.model.delete_adapter("ref")
        trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        if not trainable or any(".default." not in n for n in trainable):
            raise RuntimeError(f"unexpected trainable parameters: {[n for n in trainable if '.default.' not in n][:5]}")
        if trainer.args.device.type != device:
            raise RuntimeError(f"Trainer picked device {trainer.args.device} but {device} was resolved")
        precision = _precision_path(trainer, profile.training_dtype)
        phases["trainer_setup"] = time.time() - t0
        t_train = time.time()
        trainer.train(resume_from_checkpoint=resume_ckpt)
        train_sec = time.time() - t_train
        done_steps = int(trainer.state.global_step)
        if done_steps != steps:
            raise RuntimeError(f"optimizer steps {done_steps} != budget {steps}")

        logs = read_jsonl(log_path)
        by_step = {int(l["step"]): l for l in logs if "loss" in l}  # a re-run step (after resume) replaces its log
        losses = [by_step[k]["loss"] for k in sorted(by_step)]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise RuntimeError(f"non-finite or missing DPO loss: {losses}")
        identity: dict[str, Any] = {"checked": False, "reason": "no step-1 log (logging_steps=1 expected)"}
        first = by_step.get(1)
        if first is not None:
            lr_c = first.get("rewards/chosen", math.nan) / cfg.dpo.beta
            lr_r = first.get("rewards/rejected", math.nan) / cfg.dpo.beta
            identity = {"checked": True, "first_step_logratio_chosen": lr_c, "first_step_logratio_rejected": lr_r,
                        "atol": REFERENCE_IDENTITY_ATOL}
            if not (abs(lr_c) <= REFERENCE_IDENTITY_ATOL and abs(lr_r) <= REFERENCE_IDENTITY_ATOL):
                raise ReferenceMismatch(f"policy != reference before the first update: {identity}")

        # ---- capture in-memory result, save to staging, free, reload, compare ------------
        t0 = time.time()
        trained_state = adapter_state_cpu(trainer.model)
        # the unwrapped forward (no accelerate autocast wrapper) -- the same forward as the reload
        policy = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
        trained_lp = sequence_logps(policy, check_pairs, device, pad_id)
        staging = make_staging_dir(Path(request.output_root), cid)
        try:
            trainer.model.save_pretrained(str(staging), selected_adapters=["default"])
            adapter_config = json.loads((staging / "adapter_config.json").read_text())
            del trainer, model, base, policy
            free_memory()
            phases["save"] = time.time() - t0
            t0 = time.time()
            load_check = self._load_check(profile, staging, trained_state, trained_lp, check_pairs, device, pad_id)
            phases["reload_check"] = time.time() - t0
            if not load_check["ok"]:
                raise LoadCheckFailed(f"published adapter does not reproduce the trained model: {load_check}")
            changed = None
            if inc.adapter_path is not None:
                changed = max_abs_diff(trained_state, load_adapter_file_state(Path(inc.adapter_path)))
            final_dir = Path(request.output_root).resolve() / cid
            ref = CheckpointRef(
                checkpoint_id=cid,
                model_profile=request.model_profile,
                base_model=profile.base_model,
                base_revision=profile.base_revision,
                adapter_path=str(final_dir),
                adapter_sha256=adapter_sha256(staging, ADAPTER_FILES),
                parent_checkpoint_id=inc.checkpoint_id,
            )
            record = CheckpointRecord(
                checkpoint=ref,
                created_at=now_iso(),
                cycle=request.cycle,
                run_id=request.run_id,
                reference_checkpoint_id=inc.checkpoint_id,
                dataset_sha256=ds_sha,
                n_train_examples=len(pairs),
                optimizer_steps=done_steps,
                train_seed=request.seed,
                trainer=self.name,
                tokenizer_sha256=tok_sha,
                chat_template_sha256=tmpl_sha,
                adapter_config=adapter_config,
                training_config=tc,
                metrics={
                    "dataset": ds_info,
                    "render": report.summary(),
                    "device": dev,
                    "dtype": profile.training_dtype,
                    "precision": precision,
                    # TRL disable_dropout=True zeroes every nn.Dropout incl. LoRA dropout, so policy and
                    # precomputed reference see identical forwards; lora.dropout is recorded but inactive.
                    "effective_lora_dropout": 0.0,
                    "chat_template_kwargs": profile.chat_template_kwargs,
                    **init,
                    "adapter_max_abs_change_vs_incoming": changed,
                    "reference": ref_check,
                    "reference_cache": {"hits": cache.hits, "misses": cache.misses, "dir": str(cache.root)},
                    "reference_identity_first_step": identity,
                    "resumed_from": resume_ckpt,
                    "save_every_steps": save_every,
                    "rng_state_scope": "python, numpy, torch CPU (+CUDA) via transformers.Trainer; no MPS RNG",
                    "seeds": {"train": request.seed, "data": request.seed, "lora_init": request.seed},
                    "logs": logs,
                    "losses": losses,
                    "trained_logps": [list(v) for v in trained_lp],
                    "durations_sec": {
                        "reference_logps": round(ref_sec, 3),
                        "train": round(train_sec, 3),
                        **{k: round(v, 3) for k, v in phases.items()},
                        "total": round(time.time() - t_start, 3),
                    },
                    "versions": _versions(),
                },
                load_check=load_check,
            )
            path = publish(staging, Path(request.output_root), record)
        except BaseException:
            discard_staging(staging)
            raise
        return CheckpointRecord.model_validate(read_json(path))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_check(profile, staging: Path, trained_state, trained_lp, pairs, device: str, pad_id: int) -> dict[str, Any]:
        base = load_base_model(profile, device)
        model = load_adapter(base, staging, trainable=False)
        try:
            pdiff = max_abs_diff(adapter_state_cpu(model), trained_state)
            file_diff = max_abs_diff(load_adapter_file_state(staging), trained_state)
            lp = sequence_logps(model, pairs, device, pad_id)
        finally:
            del model, base
            free_memory()
        ldiff = max(abs(a - b) for x, y in zip(lp, trained_lp) for a, b in zip(x, y))
        ok = pdiff is not None and file_diff is not None and pdiff <= LOAD_CHECK_PARAM_ATOL and file_diff <= LOAD_CHECK_PARAM_ATOL and ldiff <= LOAD_CHECK_LOGP_ATOL
        return {
            "ok": bool(ok),
            "param_max_abs_diff": pdiff,
            "file_param_max_abs_diff": file_diff,
            "logp_max_abs_diff": ldiff,
            "param_atol": LOAD_CHECK_PARAM_ATOL,
            "logp_atol": LOAD_CHECK_LOGP_ATOL,
            "n_examples": len(pairs),
            "reloaded_logps": [list(v) for v in lp],
        }


def _precision_path(trainer: Any, dtype: str) -> dict[str, Any]:
    """Fail unless the Trainer runs without mixed precision (e.g. ACCELERATE_MIXED_PRECISION in the
    environment would re-enable accelerate's autocast forward wrapper)."""
    acc = trainer.accelerator
    mp = str(getattr(acc, "mixed_precision", "no"))
    native_amp = bool(getattr(acc, "native_amp", False))
    wrapped = "_original_forward" in getattr(trainer.model, "__dict__", {})
    info = {"weights_dtype": dtype, "trainer_mixed_precision": mp, "native_amp": native_amp,
            "forward_wrapped_by_accelerate": wrapped, "logits": "float32 (lm_head forward hook)",
            "logp_path": LOGP_PATH}
    if mp != "no" or native_amp or wrapped:
        raise TrainingRequestError(f"Trainer mixed precision must be off (one log-prob path): {info}")
    return info


def _versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for p in ("torch", "transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out
