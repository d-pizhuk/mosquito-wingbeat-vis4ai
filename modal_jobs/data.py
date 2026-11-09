"""CPU-only dataset download and preparation. No GPU is requested here."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile

from common.config import DataConfig, safe_name
from modal_jobs.runtime import (app, cpu_image, data_volume, DATA_MOUNT,
                                kaggle_secrets)


@app.function(image=cpu_image, cpu=(4.0, 4.0), memory=(8192, 8192),
              volumes={str(DATA_MOUNT): data_volume}, secrets=kaggle_secrets,
              timeout=4 * 3600, max_containers=1, retries=0, single_use_containers=True)
def prepare_data(preparation_id: str = "wingbeats-v1", data_config: dict | None = None,
                 kaggle_handle: str = "potamitis/wingbeats", upload_path: str | None = None,
                 groups_path: str | None = None) -> dict:
    from common.prepare import prepare_dataset, safe_extract_zip
    from common.io_utils import emit, file_sha256
    configuration = DataConfig.from_dict(data_config or {})
    safe_name(preparation_id)
    data_volume.reload()
    destination = DATA_MOUNT / "prepared" / preparation_id

    def uploaded(relative: str) -> Path:
        path = (DATA_MOUNT / relative).resolve()
        if not path.is_relative_to(DATA_MOUNT) or not path.exists():
            raise ValueError(f"Invalid or missing uploaded path: {relative}")
        return path

    groups = uploaded(groups_path) if groups_path else None
    group_digest = file_sha256(groups) if groups else None
    request = {"upload_path": upload_path, "kaggle_handle": kaggle_handle if not upload_path else None,
               "groups_sha256": group_digest}
    if (destination / "metadata.json").exists():
        existing = json.loads((destination / "metadata.json").read_text())
        if existing["data_config"] != configuration.to_dict() or existing["source"].get("request") != request:
            raise ValueError("This preparation_id already refers to different settings/source. Choose a new ID")
        emit("prepared_cache_reused", preparation_id=preparation_id)
        return {"preparation_id": preparation_id, **existing}
    with tempfile.TemporaryDirectory(prefix="wingbeats-raw-") as temp:
        if upload_path:
            uploaded_source = uploaded(upload_path)
            if uploaded_source.is_dir():
                raw_root = uploaded_source
            elif uploaded_source.suffix.lower() == ".zip":
                raw_root = Path(temp)
                safe_extract_zip(uploaded_source, raw_root)
            else:
                raise ValueError("Upload a dataset ZIP or an extracted dataset directory")
            source = {"kind": "uploaded", "request": request}
        else:
            import kagglehub
            emit("download_start", dataset=kaggle_handle, device="CPU only")
            try:
                raw_root = Path(kagglehub.dataset_download(kaggle_handle))
            except Exception as error:
                raise RuntimeError(
                    "Kaggle download failed. Accept any dataset terms on Kaggle, configure an optional "
                    "Modal Secret containing KAGGLE_API_TOKEN, or upload wingbeats.zip to the data Volume. "
                    "No GPU has been allocated by this function.") from error
            source = {"kind": "kaggle", "request": request, "resolved_cache_path": str(raw_root)}
        result = prepare_dataset(raw_root, destination, configuration, source=source, groups_csv=groups)
    data_volume.commit()
    return {"preparation_id": preparation_id, **result}
