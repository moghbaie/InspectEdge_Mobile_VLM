# InspectEdge — Mobile VLM fine-tuning (Colab-first)

Everything runs in **Google Colab**. The local copy is only used for writing and smoke-testing code.

## How to run
1. Open a notebook from `notebooks/` in Colab (File → Open notebook → GitHub → `moghbaie/InspectEdge_Mobile_VLM`). The setup cell clones this repo and downloads VisA itself; Drive (`MyDrive/InspectEdge`) only stores outputs.
2. Open the notebooks in order from Drive with *Open with → Google Colaboratory*, using a GPU runtime:

| Notebook | Purpose | Exit criterion |
|---|---|---|
| `notebooks/01_dataset_audit_colab.ipynb` | download + checksum VisA, build manifest, unit/data tests, stats, augmentation QA, B0 quality gate | all tests pass, split lock verified, gallery approved |
| `notebooks/02_baselines_colab.ipynb` | B1 PatchCore (+robustness), B3 detector+template report, B2 zero-shot SmolVLM | baseline table in `reports/` |
| `notebooks/03_vlm_finetune_colab.ipynb` | smoke overfit test, LoRA SFT, grid, ablations | smoke passes, checkpoints on Drive |
| `notebooks/04_evaluation_colab.ipynb` | val selection, pre-registration, locked test, robustness, gates, human review sheet | `reports/comparison_test.md` |

See `InspectEdge_Execution_Plan_v2.md` for the step-by-step plan with tests and validation gates.

## Layout
- `src/inspectedge/`: `schemas/report.py` (schema, validator, parser), `data/build_manifest.py`, `augmentation/corruptions.py`, `detection/patchcore.py`, `vlm/{prompting,sft_data}.py`, `evaluation/metrics.py`
- `scripts/`: baseline, training, evaluation and comparison entry points; `make_notebooks.py` regenerates the notebooks
- `tests/unit`, `tests/data`: run with `python -m pytest tests -q`
- `configs/`: `train_vlm.yaml`, `taxonomy_map.yaml`
- `data/licenses/data_sources.csv`: license register; `data/manifests/`: manifest + locked split hashes
