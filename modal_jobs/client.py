"""Notebook-side artifact transfer without installing the ML training stack."""
from pathlib import Path
import os
import tempfile


def download_volume_file(volume, remote_path: str, local_path: str | Path) -> Path:
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            for block in volume.read_file(remote_path):
                stream.write(block)
        os.replace(temp_name, destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return destination


def extract_run_bundle(archive: str | Path, destination: str | Path) -> Path:
    import stat
    import zipfile
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (root / item.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or ":" in item.filename or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Unsafe run bundle member: {item.filename}")
        zipped.extractall(root)
    return root
