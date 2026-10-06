# Dense NeuroAI Dataset Comparison

![Dataset comparison per subject](output_data/dataset_comparison_per_subject.png)

A systematic comparison of dense neuroAI datasets in terms of the volume and depth of neural data and cognitive states available. Built on the [`invoke`](https://www.pyinvoke.org/) task runner and [`airoh`](https://pypi.org/project/airoh/).

---

## Quick Start

```bash
uv sync
uv run invoke fetch
uv run invoke run
```

---

## Setup

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `pyproject.toml`.

---

## Design Principles

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/` Python modules and is run by `invoke` tasks. Notebooks only read results and produce figures — so they stay fast.
- **Idempotent steps.** Each `run-{name}` task checks whether its outputs already exist and skips if they do. You can call `invoke run` repeatedly while working on a later step without re-running earlier ones. To force a full rerun: `invoke clean && invoke run`.
- **Mirrored clean tasks.** Every `run-{name}` has a matching `clean-{name}` that removes only its outputs. The top-level `clean` calls them all.
- **Smoke test.** Tasks support a `--smoke` flag for a fast minimal run to verify the pipeline end-to-end.

---

## Tasks

| Task | Description |
|---|---|
| `fetch` | Initialize the `cneuromod.all` submodule (and its `cneuromod.all.statistics` submodule only), then validate all dataset YAML files in `source_data/` against the JSON schema |
| `run-tables` | Build tidy CSV tables from source YAML files plus the CNeuroMod summary row from `cneuromod.all.statistics` |
| `run-figures` | Execute notebooks, saving figures to `output_data/` |
| `run` | Full pipeline: `fetch` → `run-tables` → `run-figures` |
| `clean-{name}` | Remove outputs of one specific step |
| `clean` | Remove all generated outputs from `output_data/` |

Use `uv run invoke --list` or `uv run invoke --help <task>` for details.

---

## Data

- **Source data**: see [`source_data/CONTENT.md`](source_data/CONTENT.md)
- **Output data**: see [`output_data/CONTENT.md`](output_data/CONTENT.md)

---

## Folder Structure

| Folder / File | Description |
|---|---|
| `analysis/` | Pure Python analysis logic, called by invoke tasks |
| `notebooks/` | Jupyter notebooks for visualization (one per figure/table group) |
| `source_data/` | Manually curated inputs — see [`source_data/CONTENT.md`](source_data/CONTENT.md) |
| `output_data/` | Generated figures and tables — see [`output_data/CONTENT.md`](output_data/CONTENT.md) |
| `tasks.py` | Invoke task definitions |
| `invoke.yaml` | Path configuration |
