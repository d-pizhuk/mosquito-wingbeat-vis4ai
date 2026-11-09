"""Single-A100 jobs, called from notebooks or the Modal CLI."""
from __future__ import annotations
from pathlib import Path
import time

from common.config import TrainConfig, safe_name
from modal_jobs.runtime import (app, data_volume, run_volume, DATA_MOUNT,
                                RUN_MOUNT, GPU_RESOURCES)


def _stage(preparation_id: str) -> Path:
    import torch
    from common.load_data import stage_locally
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The remote worker has no CUDA GPU")
    safe_name(preparation_id)
    data_volume.reload()
    run_volume.reload()
    return stage_locally(DATA_MOUNT / "prepared" / preparation_id)


@app.function(**GPU_RESOURCES, timeout=1800)
def benchmark(preparation_id: str, train_config: dict, measured_steps: int = 30) -> dict:
    from common.io_utils import atomic_json
    from training.benchmark import benchmark_training
    config = TrainConfig.from_dict(train_config)
    data = _stage(preparation_id)
    result = benchmark_training(data, config, device="cuda", measured_steps=measured_steps)
    output = RUN_MOUNT / "benchmarks" / f"{safe_name(preparation_id)}-{int(time.time())}.json"
    atomic_json(output, {"configuration": train_config, **result})
    run_volume.commit()
    return result


@app.function(**GPU_RESOURCES, timeout=8 * 3600)
def train(preparation_id: str, run_id: str, train_config: dict, resume: bool = False) -> dict:
    from training.engine import run_training
    from common.io_utils import atomic_json
    entry_started = time.monotonic()
    config = TrainConfig.from_dict(train_config)
    safe_name(run_id)
    data = _stage(preparation_id)
    run = RUN_MOUNT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "modal_run.json", {"preparation_id": preparation_id, "run_id": run_id,
                                        "gpu": "A100-80GB", "cpu_cores": 4, "host_memory_gib": 16})
    return run_training(data, run, config, resume=resume, device="cuda",
                        commit=run_volume.commit, entry_started=entry_started)


@app.function(**GPU_RESOURCES, timeout=2 * 3600)
def evaluate(preparation_id: str, run_id: str, split: str = "test",
             limit: int | None = None, batch_size: int = 128) -> dict:
    from common.inference import evaluate_checkpoint
    safe_name(run_id)
    if split not in {"validation", "test"}:
        raise ValueError("Cloud evaluation accepts validation or test only")
    data = _stage(preparation_id)
    run = RUN_MOUNT / "runs" / run_id
    checkpoint = run / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("No best.pt yet. At least one validation pass must finish first")
    result = evaluate_checkpoint(checkpoint, data, run / "evaluation" / split,
                                 split=split, limit=limit, batch_size=batch_size, device="cuda",
                                 max_seconds=3600)
    run_volume.commit()
    return result


@app.function(**GPU_RESOURCES, timeout=1800)
def export_attention(preparation_id: str, run_id: str, sample_ids: list[str]) -> dict:
    from common.inference import attention_examples
    safe_name(run_id)
    data = _stage(preparation_id)
    run = RUN_MOUNT / "runs" / run_id
    result = attention_examples(run / "best.pt", data, sample_ids,
                                run / "phase2_attention", device="cuda")
    run_volume.commit()
    return result
