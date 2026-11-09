"""Import this module from Jupyter, or run: modal run -m modal_jobs.app ..."""
from common.config import DataConfig, TrainConfig
from modal_jobs.runtime import app, data_volume, run_volume
from modal_jobs.data import prepare_data
from modal_jobs.gpu import benchmark, train, evaluate, export_attention
from modal_jobs.artifacts import package_run, status


@app.local_entrypoint()
def main(action: str = "status", run_id: str = "r50-cbam-full-001",
         preparation_id: str = "wingbeats-v1", config: str = "configs/full.json",
         resume: bool = False):
    import json
    if action == "prepare":
        result = prepare_data.remote(preparation_id, DataConfig().to_dict())
    elif action == "benchmark":
        result = benchmark.remote(preparation_id, TrainConfig.load(config).to_dict())
    elif action == "train":
        result = train.remote(preparation_id, run_id, TrainConfig.load(config).to_dict(), resume)
    elif action == "evaluate":
        result = evaluate.remote(preparation_id, run_id)
    elif action == "package":
        result = package_run.remote(run_id)
    elif action == "status":
        result = status.remote(run_id)
    else:
        raise ValueError("action must be prepare, benchmark, train, evaluate, package or status")
    print(json.dumps(result, indent=2))
