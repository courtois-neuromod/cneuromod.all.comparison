from pathlib import Path
from invoke import task


@task
def fetch(c):
    """Initialize/update submodules and validate dataset YAML files against the cneuromod.all schema."""
    from airoh.utils import ensure_submodule
    import json
    import yaml
    import jsonschema

    ensure_submodule(c, "source_data/cneuromod", recursive=False)
    # Only the statistics submodule is needed (for the CNeuroMod summary row);
    # never init cneuromod.all recursively — it pulls every dataset.
    c.run("git -C source_data/cneuromod submodule update --init "
          "analysis/cneuromod.all.statistics")

    source_dir = Path(c.config.get("source_data_dir"))
    schema_file = source_dir / "cneuromod" / "docs" / "schema.json"
    with open(schema_file) as f:
        schema = json.load(f)

    # Allow tasks.contrasts to be either a plain integer (cneuromod legacy) or
    # a {total, per_subject} object (used when the two values differ).
    schema["properties"]["tasks"]["properties"]["contrasts"] = {
        "oneOf": [
            {"type": "integer"},
            {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "per_subject": {"type": "integer"},
                },
                "required": ["total", "per_subject"],
                "additionalProperties": False,
            },
        ]
    }

    yaml_files = sorted(source_dir.glob("*.yaml"))
    if not yaml_files:
        print("No dataset YAML files found in source_data/ — nothing to validate.")
        return

    errors = []
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        try:
            jsonschema.validate(data, schema)
            print(f"  OK  {yaml_file.name}")
        except jsonschema.ValidationError as e:
            print(f"  ERR {yaml_file.name}: {e.message}")
            errors.append(yaml_file.name)

    if errors:
        raise SystemExit(f"Validation failed: {', '.join(errors)}")
    print(f"All {len(yaml_files)} dataset(s) valid.")


@task
def run_tables(c):
    """Generate tidy summary tables from the dataset YAML files."""
    from analysis.tables import (
        build_tidy_table,
        COLUMN_GROUPS_PER_SUBJECT,
        COLUMN_GROUPS_TOTAL,
    )

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cneuromod_yaml = Path(c.config.get("cneuromod_summary"))
    if not cneuromod_yaml.exists():
        raise SystemExit(f"{cneuromod_yaml} not found — run `invoke fetch` first "
                         "(it is produced and tracked by cneuromod.all.statistics).")

    for scope, groups in [
        ("per_subject", COLUMN_GROUPS_PER_SUBJECT),
        ("total", COLUMN_GROUPS_TOTAL),
    ]:
        df = build_tidy_table(source_dir, groups, extra_yaml_files=[cneuromod_yaml])
        out_path = output_dir / f"datasets_tidy_{scope}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path.name}")


@task(pre=[run_tables])
def run_figures(c):
    """Generate figures from the dataset YAML files using notebooks."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()

    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_tables, run_figures])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    run_figures(c)


@task
def clean_notebooks(c):
    """Remove notebook outputs from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.png")
    clean_folder(c, "output_data_dir", "*.csv")


@task(pre=[clean_notebooks])
def clean(c):
    """Remove all computed outputs."""
    pass
