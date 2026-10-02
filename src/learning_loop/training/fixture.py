"""`trainer: fixture` — a clearly labeled pseudo-trainer for orchestration tests.

It has the same request/publication/lineage/reference semantics as the TRL DPO
trainer but no model: the "adapter" is `fixture_adapter.json`, a short vector
whose values are a deterministic function of the incoming vector, the dataset
hash and the step index. Cycle tests can therefore check that each cycle
continues the incoming weights and records the incoming checkpoint as its
reference, without torch. It never produces a usable model.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..interfaces import TrainRequest
from ..records import CheckpointRecord, CheckpointRef
from ..storage import atomic_write_json, now_iso, read_json
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
    resolve_training_config,
)

FIXTURE_ADAPTER = "fixture_adapter.json"
FIXTURE_KIND = "fixture_pseudo_adapter"
N_WEIGHTS = 8


class SimulatedInterruption(RuntimeError):
    """Raised by the test hook that interrupts training after a given optimizer step."""


def _weights_sha(w: list[float]) -> str:
    return hashlib.sha256(json.dumps(w, separators=(",", ":")).encode()).hexdigest()


def _delta(dataset_sha: str, step: int, i: int) -> float:
    h = hashlib.sha256(f"{dataset_sha}:{step}:{i}".encode()).digest()
    return round((int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 0.01, 12)


def load_fixture_weights(ref: CheckpointRef) -> list[float]:
    if ref.adapter_path is None:
        return [0.0] * N_WEIGHTS
    p = Path(ref.adapter_path) / FIXTURE_ADAPTER
    if not p.exists():
        raise TrainingRequestError(f"incoming checkpoint {ref.checkpoint_id} is not a fixture adapter ({p} missing)")
    if ref.adapter_sha256 and adapter_sha256(p.parent, (FIXTURE_ADAPTER,)) != ref.adapter_sha256:
        raise TrainingRequestError(f"incoming fixture adapter {p} does not match its recorded sha256")
    data = read_json(p)
    if data.get("kind") != FIXTURE_KIND:
        raise TrainingRequestError(f"{p} is not a {FIXTURE_KIND}")
    return [float(x) for x in data["weights"]]


class FixtureTrainer:
    name = "fixture"

    def __init__(self, _test_interrupt_after_step: int | None = None):
        self._interrupt_after = _test_interrupt_after_step

    def train(self, request: TrainRequest, work_dir: Path) -> CheckpointRecord:
        t0 = time.time()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg = resolve_training_config(request.training_config, self.name)
        if request.model_profile != request.incoming.model_profile:
            raise TrainingRequestError("request.model_profile != incoming.model_profile")
        examples, ds_sha, ds_info = load_preference_dataset(request.dataset_dir)
        if not examples:
            raise NoTrainableExamples(f"{request.dataset_dir}: no preference examples")
        tc = cfg.model_dump()
        cid = derive_checkpoint_id(request.run_id, request.cycle, ds_sha, request.incoming.checkpoint_id, tc, request.seed, self.name)
        done = existing_publication(Path(request.output_root), cid, ds_sha)
        if done is not None:
            return done

        # Stage state: resume from this stage's own state file only.
        req_sha = request_sha256(request.model_dump())
        state_path = work_dir / "fixture_state.json"
        incoming_w = load_fixture_weights(request.incoming)
        resumed_from = None
        if state_path.exists():
            st = read_json(state_path)
            if st["request_sha256"] != req_sha:
                raise TrainingRequestError(f"{state_path} belongs to a different training request; use a fresh work dir")
            step, w = int(st["step"]), [float(x) for x in st["weights"]]
            resumed_from = step
        else:
            step, w = 0, list(incoming_w)
        while step < cfg.optimizer_steps:
            w = [round(x + _delta(ds_sha, step, i), 12) for i, x in enumerate(w)]
            step += 1
            atomic_write_json(state_path, {"request_sha256": req_sha, "step": step, "weights": w})
            if self._interrupt_after is not None and step == self._interrupt_after and resumed_from is None:
                raise SimulatedInterruption(f"fixture training interrupted after step {step}")

        staging = make_staging_dir(Path(request.output_root), cid)
        try:
            payload = {
                "kind": FIXTURE_KIND,
                "label": "FIXTURE pseudo-adapter for orchestration tests; not model weights",
                "weights": w,
                "initial_weights_sha256": _weights_sha(incoming_w),
                "reference_checkpoint_id": request.incoming.checkpoint_id,
                "dataset_sha256": ds_sha,
                "optimizer_steps": step,
            }
            (staging / FIXTURE_ADAPTER).write_text(json.dumps(payload, indent=2) + "\n")
            # load check: re-read what will be published and compare exactly
            reread = read_json(staging / FIXTURE_ADAPTER)["weights"]
            load_ok = reread == w
            if not load_ok:
                raise RuntimeError("fixture adapter reload mismatch")
            final_dir = Path(request.output_root).resolve() / cid
            ref = CheckpointRef(
                checkpoint_id=cid,
                model_profile=request.model_profile,
                base_model=request.incoming.base_model,
                base_revision=request.incoming.base_revision,
                adapter_path=str(final_dir),
                adapter_sha256=adapter_sha256(staging, (FIXTURE_ADAPTER,)),
                parent_checkpoint_id=request.incoming.checkpoint_id,
            )
            record = CheckpointRecord(
                checkpoint=ref,
                created_at=now_iso(),
                cycle=request.cycle,
                run_id=request.run_id,
                reference_checkpoint_id=request.incoming.checkpoint_id,
                dataset_sha256=ds_sha,
                n_train_examples=len(examples),
                optimizer_steps=step,
                train_seed=request.seed,
                trainer=self.name,
                adapter_config={"kind": FIXTURE_KIND, "n_weights": N_WEIGHTS},
                training_config=tc,
                metrics={
                    "fixture": True,
                    "dataset": ds_info,
                    "initial_weights_sha256": _weights_sha(incoming_w),
                    "initial_weights_match_incoming": True,
                    "final_weights_sha256": _weights_sha(w),
                    "resumed_from_step": resumed_from,
                    "duration_sec": round(time.time() - t0, 4),
                },
                load_check={"ok": load_ok, "method": "exact JSON reload", "tolerance": 0.0},
            )
            path = publish(staging, Path(request.output_root), record)
        except BaseException:
            discard_staging(staging)
            raise
        return CheckpointRecord.model_validate(read_json(path))


def base_checkpoint_ref(profile_name: str, checkpoint_id: str = "base") -> CheckpointRef:
    """The incoming reference for cycle 0: the profile's base model, no adapter."""
    from ..config import load_model_profile

    prof = load_model_profile(profile_name)
    return CheckpointRef(
        checkpoint_id=checkpoint_id,
        model_profile=profile_name,
        base_model=prof.base_model,
        base_revision=prof.base_revision,
    )


__all__: list[Any] = ["FixtureTrainer", "SimulatedInterruption", "base_checkpoint_ref", "load_fixture_weights"]
