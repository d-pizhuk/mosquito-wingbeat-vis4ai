"""Cloud resources and images. Importing this file does not launch a GPU."""
from pathlib import Path
import os
import modal

APP_NAME = "mosquito-wingbeat-vis4ai"
DATA_VOLUME_NAME = "wingbeat-data"
RUN_VOLUME_NAME = "wingbeat-runs"
DATA_MOUNT = Path("/datasets")
RUN_MOUNT = Path("/artifacts")
GPU = "A100-80GB"
GPU_CPU_CORES = 4.0
GPU_HOST_MEMORY_MIB = 16384

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)
volumes = {str(DATA_MOUNT): data_volume, str(RUN_MOUNT): run_volume}

# Optional only: the public Kaggle download normally does not need a secret.
# Set WINGBEAT_KAGGLE_SECRET before importing this module if consent/auth is required.
secret_name = os.environ.get("WINGBEAT_KAGGLE_SECRET")
kaggle_secrets = [modal.Secret.from_name(secret_name)] if secret_name else []

cpu_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("numpy==2.3.5", "scipy==1.17.0", "pandas==2.2.3", "soundfile==0.13.1",
                 "scikit-learn==1.8.0", "kagglehub==1.0.3")
    .env({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
          "OPENBLAS_NUM_THREADS": "1", "KAGGLEHUB_CACHE": "/tmp/kagglehub"})
)
gpu_base = (
    cpu_base
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("librosa==0.11.0")
    .env({"TORCH_HOME": "/opt/torch-cache", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    # Cache ImageNet weights at image-build time, without allocating a GPU.
    .run_commands("python -c 'from torchvision.models import resnet50, ResNet50_Weights; "
                  "resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)' ")
)
# Add source after image build commands. Heavy libraries are imported INSIDE jobs.
cpu_image = cpu_base.add_local_python_source("common", "training", "modal_jobs")
gpu_image = gpu_base.add_local_python_source("common", "training", "modal_jobs")

GPU_RESOURCES = dict(image=gpu_image, gpu=GPU, cpu=(GPU_CPU_CORES, GPU_CPU_CORES),
                     memory=(GPU_HOST_MEMORY_MIB, GPU_HOST_MEMORY_MIB),
                     volumes=volumes, max_containers=1, retries=0, single_use_containers=True)
