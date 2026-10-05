# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project systematically compares dense neuroAI datasets in terms of the volume and depth of neural data and cognitive states available. It is built on the [`invoke`](https://www.pyinvoke.org/) task runner. The `airoh` pip package provides reusable invoke tasks; this repo customizes them via `tasks.py` and `invoke.yaml`.

**Package manager:** `uv` (see `pyproject.toml`).

**Pipeline:** `fetch` validates all YAML files in `source_data/` against the JSON Schema at `source_data/cneuromod/docs/schema.json`; `run-tables` runs `analysis/tables.py` to produce `output_data/datasets_tidy_*.csv`; `run-figures` executes notebooks in `notebooks/` (reading the tidy CSVs) to produce figures in `output_data/`. All source data is manually curated and version-controlled via git.

## Dataset Assets

Each dataset has two files in `source_data/`:

- `<name>.yaml` — structured data entry, validated against `source_data/schema.json`. Only populate fields that are relevant to the dataset (e.g. omit `neuroimaging.meg` entirely if the dataset has no MEG).
- `<name>.md` — markdown sidecar that justifies every value in the YAML with direct quotes from the corresponding publication(s) or official documentation.

The markdown sidecar should:
- Cover each field present in the YAML
- Justify every number with a direct quote from the publication or documentation
- Cite the source of each quote (paper title/DOI or URL)
- Flag any values that could not be confirmed or are approximate

When asked to work on a dataset entry:
1. Identify which schema fields are relevant to the dataset
2. Search for the paper(s) and/or documentation to find supporting quotes
3. Draft or update `source_data/<name>.yaml` with only the relevant fields
4. Draft or update `source_data/<name>.md` with quoted evidence for each field
5. Flag any fields where information could not be found or is ambiguous

## Setup

```bash
# uv (recommended):
uv sync

# pip:
pip install -r requirements.txt
```

## Common Commands

With `uv`:
```bash
uv run invoke fetch           # Validate dataset YAML files against schema
uv run invoke run             # Full pipeline (fetch → run-notebooks)
uv run invoke run-notebooks   # Execute notebooks, save figures to output_data/
uv run invoke clean           # Remove output_data/ contents
uv run invoke --list          # Show all available tasks
```

Without `uv` (activate your environment first):
```bash
invoke fetch              # Validate dataset YAML files against schema
invoke run                # Full pipeline
invoke run-notebooks      # Execute notebooks, save figures to output_data/
invoke clean              # Remove output_data/ contents
invoke --list             # Show all available tasks
```

## Architecture

**Execution flow:** `invoke run` triggers the project's analysis pipeline via `pre=` dependencies declared in `tasks.py`. The three permanent tasks — `fetch`, `run`, `clean` — are always present; intermediate steps are project-specific.

- `invoke.yaml` — all path config (`output_data_dir`, `source_data_dir`, `notebooks_dir`)
- `tasks.py` — project-specific invoke tasks; imports reusable tasks from `airoh.utils`
- `analysis/tables.py` — defines `COLUMN_GROUPS` (field registry with labels, dotpaths, units, colors) and `build_tidy_table(source_dir)` which produces the long-format DataFrame; run via `run-tables`
- `source_data/cneuromod/docs/schema.json` — authoritative JSON Schema for dataset YAML files; edit here to add new modalities or fields
- `notebooks/` — Jupyter notebooks executed by `run_notebooks` via `airoh.utils.run_notebooks`; notebooks receive `OUTPUT_DATA_DIR` as an environment variable and read `datasets_tidy.csv` from it
- `source_data/CONTENT.md` and `output_data/CONTENT.md` — authoritative docs for what each data folder contains; update these when data assets change, do not duplicate their content elsewhere

**Analysis vs. notebooks:** Heavy computation belongs in `analysis/` Python code, invoked by `run-{name}` tasks, which write results to `output_data/`. Notebooks are for visualization only — they read from `output_data/` and produce figures. This keeps notebooks fast and focused.

**Idempotent tasks:** Each `run-{name}` task must check whether its outputs already exist and skip execution if they do. This means `invoke run` can be called repeatedly during development of a later step — earlier steps are skipped automatically. To force a full rerun, call `invoke clean` first, then `invoke run`.

**Task naming conventions:**
- Analysis tasks are named `run-{name}` (e.g. `run-tables`, `run-figures`).
- Cleaning tasks mirror them: `clean-{name}` removes only the outputs of the corresponding step.
- The top-level `clean` task calls all `clean-{name}` tasks in sequence.
- The top-level `run` task wires all steps together via `pre=` chains in `tasks.py`.

**Task parameters:** `run-{name}` tasks should expose chunk or subset parameters (e.g. a dataset name) so that individual pieces can be rerun in isolation. They should also support a `smoke` flag for a fast minimal run useful for testing the pipeline end-to-end without running the full analysis.

**Adding a new schema field:** edit `source_data/cneuromod/docs/schema.json`, then update any existing YAML files that should carry the new field. Note: that file lives inside a git submodule — changes must go through the `cneuromod.all` repository.

**Adding a new analysis step:** add a function to `analysis/`, add a `run-{name}` task and a matching `clean-{name}` task in `tasks.py`, wire both into the top-level `run` and `clean` tasks via `pre=` chains, and create or extend a notebook in `notebooks/` for visualization.

**Evolving CLAUDE.md:** As the project grows, update this file to document the specific analysis steps, data sources, and decisions for this project. The airoh conventions above (idempotent tasks, `run-{name}`/`clean-{name}` pairs, analysis-in-code/visualization-in-notebooks) apply for the lifetime of the project.

**README.md** is the user-facing documentation for this project. Any structural or workflow changes — new tasks, renamed folders, updated commands, new dependencies — must be reflected there. For data folder contents, point to `source_data/CONTENT.md` and `output_data/CONTENT.md` rather than duplicating their content inline.

## Schema field semantics

The schema defines two field types:

- **`recording_hours`** (`total_h`, `per_subject_h`): used for neural recordings and physiology — modalities where repetition is inherent and there is no notion of "unique content" (e.g. fMRI, EEG, EDA).
- **`stimulus_modality`** (`total_unique`, `per_subject_unique`, `total_with_repetition`, `per_subject_with_repetition`): used for all stimulus and task fields — both `passive.*` and `active.controlled`.

**Unique vs. with_repetition:**
- `unique` counts the unique stimulus/task content once, excluding any repetition of identical content whether within or across subjects. Example: 10 subjects each doing the same 10-minute controlled task → `total_unique = 10 min`.
- `with_repetition` counts all exposures including repetitions. Same example → `total_with_repetition = 100 min`.

**Rules for computing `unique` in controlled tasks (`active.controlled`):**
1. If the same experimental paradigm (same condition set) is repeated across sessions for the same subject, count it once.
2. If different subjects receive different subsets of tasks that share the same task environment (e.g. fractional subtask designs), treat the per-subject subset as the unique content — do not sum across subtask variants.
3. When all subjects run the same task environment, `total_unique = per_subject_unique` (no unique content is added by running more subjects).

**`source_data/cneuromod` is a git submodule** pointing to a separate project (`cneuromod.all`). Never modify any files inside `source_data/cneuromod/` — changes must go through that project's own repository. The submodule's `dataset_info.yaml` files use their own schema and are read by `aggregate_cneuromod_yaml` in `analysis/tables.py`. The per-dataset CNeuroMod comparison (tidy tables + bubble charts per CNeuroMod dataset) lives in the `cneuromod.all.statistics` repository, not here.
