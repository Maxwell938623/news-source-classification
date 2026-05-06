# `logs/`

Training and pipeline run logs. **Not tracked in Git** (matched by `*.log` in `.gitignore`) — they are large, regenerable, and would otherwise add several MB of churn per run.

Each pipeline / training script writes its own log here. Filenames follow the pattern `<script_name>.log`, for example:

* `collect_urls.log`, `scrape.log`, `preprocess.log`, `split.log` — data pipeline
* `train_baseline.log`, `train_best_model.log`, `train_<family>.log` — classical models
* `train_transformer_<model>.log` — HuggingFace fine-tunes
* `evaluate.log` — held-out test scoring

To regenerate logs locally, just rerun the corresponding script (see `../src/README.md` for the pipeline order, or `../README.md` § Quick Reproduction for the headline commands).
