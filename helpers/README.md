# `helpers/`

Course-provided handouts, templates, and a small repacking utility we wrote.

| File | Origin | Purpose |
|---|---|---|
| `CIS 5190 Final Project Descriptions.pdf` | Course staff | Full project handout (basic + exploratory requirements, §1.4) |
| `Project_submission.pdf` | Course staff | Leaderboard / Hugging Face submission contract (function signatures, I/O formats) |
| `eval_project_b.py` | Course staff | Local sanity-check evaluator that mimics the leaderboard backend; run against any package under [`../submission/`](../submission/) |
| `model_template.py` | Course staff | Reference `Model.predict` skeleton (we extend this in our submission packages) |
| `preprocess_template.py` | Course staff | Reference `prepare_data(input_csv) -> (X, y)` skeleton |
| `url_with_headlines.csv` | Course staff | Helper CSV mapping the released starter URLs to scraped headlines (used as one of our local fallbacks) |
| `repack_gbdt_model_pt.py` | Ours | Small utility that wraps a joblib pipeline into a torch `state_dict` so the leaderboard backend's `torch.load` machinery accepts it. Used to regenerate `submission/gbdt_model/model.pt` |

Local sanity check:

```bash
python helpers/eval_project_b.py --submission_dir ../submission/
```
