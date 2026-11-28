"""Compara conjuntos de atributos e tamanho de ensemble em diretório isolado."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from scripts.evaluate_walk_forward import evaluate, read_records
from scripts.train import EpisenseTrainer

HORIZONS = [1, 2, 4, 8]
CASE_FEATURES = [
    "sin_semana", "cos_semana",
    *[f"log_casos_lag{k}" for k in (1, 2, 3, 4, 8, 12, 52, 53)],
    *[f"log_casos_rolling_mean_{k}" for k in (3, 4, 8, 12)],
    "log_casos_yoy52", "log_casos_yoy53",
]
VARIANTS = {
    "full_wolbachia_one_seed": (1, None),
    "full_one_seed": (1, None),
    "cases_wolbachia_one_seed": (1, CASE_FEATURES + ["wolbachia_cg_stage"]),
    "cases_one_seed": (1, CASE_FEATURES),
    "full_five_seeds": (5, None),
    "seed_123": ((123,), None),
    "seed_456": ((456,), None),
    "seed_789": ((789,), None),
    "seed_999": ((999,), None),
    "two_seeds_42_123": ((42, 123), None),
}


def evaluation_cohort_hash(artifact, report, *, decimals=None):
    records = read_records(artifact, report["label_maturity_weeks"],
                           revised=report.get("scope") == "retrospective_revised_inputs")
    cohort = sorted(
        (fold, origin, horizon, row["target"], row["actual"])
        for (fold, origin, horizon), row in records.items()
        if row["available"] <= report["evaluation_cutoff_se"]
    )
    if len(cohort) != report["n_scored"] or not all(np.isfinite(row[-1]) for row in cohort):
        raise RuntimeError("Coorte experimental inconsistente")
    if decimals is not None:
        # expm1(log1p(casos)) diverge no último bit entre plataformas.
        cohort = [(*row[:-1], round(row[-1], decimals)) for row in cohort]
    return hashlib.sha256(json.dumps(cohort).encode()).hexdigest()


def run_variant(name, base, config, output, horizons, *, variant=None, workers=None):
    cfg = deepcopy(config)
    selection, features = VARIANTS[name] if variant is None else variant
    cfg["targets"].update(horizons=horizons, target_cols=[f"target_h{h}" for h in horizons])
    seeds = cfg["model"]["ensemble"]["seeds"][:selection] if isinstance(selection, int) else list(selection)
    num_seeds = len(seeds)
    cfg["model"]["ensemble"].update(seeds=seeds, n_seeds=len(seeds))
    cfg["model"]["hyperparameters"].update(num_threads=1, n_jobs=1)
    cfg["training"]["parallel_horizons"].update(enabled=True, n_jobs=workers or len(horizons), worker_threads=1)
    if name in {"full_wolbachia_one_seed", "cases_wolbachia_one_seed"}:
        if str(cfg["project"]["geocode"]) != "5002704":
            raise ValueError("Calendário Wolbachia específico de Campo Grande")
        cfg["features"]["expand_wolbachia_campo_grande"] = True
    if features is not None:
        cfg["training"]["feature_allowlist"] = features
    directory = output / name
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
    os.environ["EPISENSE_MODELS_DIR"] = str(directory)
    start = time.monotonic()
    EpisenseTrainer(cfg).train_all_horizons(base.copy())
    seconds = time.monotonic() - start
    artifact = json.loads((directory / "validation_predictions.json").read_text())
    report, _, _ = evaluate(artifact, base, cfg)
    if not report["training_embargo_verified"] or set(report["by_horizon"]) != set(horizons):
        raise RuntimeError(f"Validação experimental incompleta: {name}")
    if report["n_eligible"] != report["n_scored"]:
        raise RuntimeError(f"Nem todas as linhas elegíveis foram pontuadas: {name}")

    cohort_digest = evaluation_cohort_hash(artifact, report)
    models = list(directory.glob("lgbm_*.pkl"))
    if len(models) != len(horizons) * 5 * num_seeds:
        raise RuntimeError(f"Número inesperado de modelos em {name}: {len(models)}")
    result = {
        "variant": name,
        "seeds": seeds,
        "feature_allowlist": features,
        "cohort_sha256": cohort_digest,
        "model_files": len(models),
        "model_bytes": sum(path.stat().st_size for path in models),
        "train_seconds": seconds,
        "evaluation": report,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--base", type=Path, default=ROOT / "data/processed/episense_base.csv")
    parser.add_argument("--revised", action="store_true",
                        help="Experimento retrospectivo de desenvolvimento com dados revisados")
    parser.add_argument("--all-horizons", action="store_true")
    args = parser.parse_args()
    horizons = list(range(1, 9)) if args.all_horizons else HORIZONS
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "logs"):
        parser.error("--output deve ficar em logs/ para isolar os artefatos da produção")
    output.mkdir(parents=True)
    snapshot = args.base.read_bytes()
    snapshot_path = output / "training_base.csv"
    snapshot_path.write_bytes(snapshot)
    os.environ["EPISENSE_TRAINING_BASE"] = str(snapshot_path)
    base = pd.read_csv(snapshot_path, dtype={"SE": str})
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    if args.revised:
        cfg["training"]["historical_inputs"] = "revised"
    results = [run_variant(name, base, cfg, output, horizons)
               for name in (args.variant or ("full_one_seed", "cases_one_seed", "full_five_seeds"))]
    if len({result["cohort_sha256"] for result in results}) != 1:
        raise RuntimeError("Variantes avaliadas em coortes diferentes")
    summary = {"base_sha256": hashlib.sha256(snapshot).hexdigest(),
               "horizons": horizons, "variants": results}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
