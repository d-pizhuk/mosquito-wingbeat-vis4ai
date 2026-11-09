## Dataset storage

We use the [WINGBEATS dataset](https://www.kaggle.com/datasets/potamitis/wingbeats) from Kaggle.

The `prepare_data` function in `modal_jobs/data.py` downloads and prepares the dataset using Modal's CPUs. The prepared data is saved in the persistent `wingbeat-data` Volume under `/datasets/prepared/<preparation_id>/`.

The following files are created:

- `waveforms.npy` — processed audio recordings.
- `manifest.csv` — information about each recording, including its species, file path, and dataset split.
- `duplicates.csv` — recordings that were removed because they were duplicates.
- `metadata.json` — dataset settings, class counts, and other information.

The dataset is split into **70% training, 10% validation, and 20% testing**. The split is based on individual recordings, so recordings from the same mosquito or recording session may appear in different splits.