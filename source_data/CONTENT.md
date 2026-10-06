# Source Data

All source data is manually curated and version-controlled via git.

- `cneuromod/` — git submodule of [cneuromod.all](https://github.com/courtois-neuromod/cneuromod.all), initialized non-recursively by `invoke fetch`. Only two files are read from it:
  - `docs/schema.json` — JSON Schema defining all possible fields for a dataset entry; every `<name>.yaml` here is validated against it.
  - `analysis/cneuromod.all.statistics/output_data/cneuromod_summary.yaml` — CNeuroMod as one schema-compatible entry (all CNeuroMod datasets summed), produced by [cneuromod.all.statistics](https://github.com/courtois-neuromod/cneuromod.all.statistics) and fed to `run-tables` as the CNeuroMod row. `fetch` initializes only this nested submodule. Path set by `cneuromod_summary` in `invoke.yaml`.
- `<name>.yaml` — one YAML file per (non-CNeuroMod) dataset, with only the relevant fields populated. Validated against the schema by `invoke fetch`.
- `<name>.md` — markdown sidecar per dataset, justifying each value with direct quotes from the corresponding publication(s) or official documentation.
