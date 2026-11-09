"""Cheap CPU access to progress and portable run bundles."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile

from common.config import safe_name
from modal_jobs.runtime import app, cpu_image, run_volume, RUN_MOUNT


@app.function(image=cpu_image, cpu=(0.25, 0.25), memory=(512, 512),
              volumes={str(RUN_MOUNT): run_volume}, timeout=60, retries=0)
def status(run_id: str) -> dict:
    safe_name(run_id)
    run_volume.reload()
    root = RUN_MOUNT / "runs" / run_id
    if not root.exists():
        return {"status": "not_started", "run_id": run_id}
    result = {"run_id": run_id}
    for name in ("status", "failure", "modal_run"):
        path = root / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    if (root / "history.csv").exists():
        result["history_csv"] = (root / "history.csv").read_text()
    return result


@app.function(image=cpu_image, cpu=(1.0, 1.0), memory=(1024, 1024),
              volumes={str(RUN_MOUNT): run_volume}, timeout=1200, retries=0)
def package_run(run_id: str, include_resume_checkpoint: bool = False) -> dict:
    safe_name(run_id)
    run_volume.reload()
    root = RUN_MOUNT / "runs" / run_id
    if not root.exists():
        raise FileNotFoundError(f"Unknown run: {run_id}")
    exports = RUN_MOUNT / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    destination = exports / f"{run_id}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    # No CPU-intensive compression of large model weights.
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in {".tmp", ".partial"}:
                continue
            if path.name == "last.pt" and not include_resume_checkpoint:
                continue
            bundle.write(path, path.relative_to(root.parent))
    temporary.replace(destination)
    run_volume.commit()
    return {"volume_path": destination.relative_to(RUN_MOUNT).as_posix(),
            "bytes": destination.stat().st_size, "includes_last_pt": include_resume_checkpoint}
