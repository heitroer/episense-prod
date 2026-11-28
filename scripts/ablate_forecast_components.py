"""Compara ablações isoladas nos oito horizontes sobre uma cópia da mesma base."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

from benchmark_model_variants import ROOT, EpisenseTrainer, evaluate, evaluation_cohort_hash, run_variant

FEATURE_VARIANTS = ("all_features", "no_weather", "no_case_lags", "no_regime_features")
# Apenas marcadores presentes em nomes de medições e de seus derivados.
WEATHER = re.compile(r"temp|umid|precip|wind|vento|chuva", re.IGNORECASE)
CASE_LAG = re.compile(r"^(?:log_lag|lag\d+_x_is_|log_casos_(?:rate_)?(?:weighted_)?lag)")


def feature_list(base, config):
    trainer = EpisenseTrainer(config)
    frame = trainer.prepare_data(base)
    candidates, _, _, _ = trainer.prepare_horizon_data(frame, 1)
    return list(candidates.columns)


def train_variant(output, name):
    base_path = output / "training_base.csv"
    os.environ["EPISENSE_TRAINING_BASE"] = str(base_path)
    base = pd.read_csv(base_path, dtype={"SE": str})
    config = yaml.safe_load((output / "config.yaml").read_text())
    columns = feature_list(base, config)
    if name == "no_regime_features":
        config_without = deepcopy(config)
        config_without["features"]["expand_regime_conditional"] = False
        keep = set(feature_list(base, config_without))
        removed = [col for col in columns if col not in keep]
    elif name in ("no_weather", "no_case_lags"):
        pattern = WEATHER if name == "no_weather" else CASE_LAG
        removed = [col for col in columns if pattern.search(col)]
    else:
        removed = []
    if not removed and name != "all_features":
        raise ValueError(f"A ablação {name} não removeu atributos")
    dropped = set(removed)
    allow = [col for col in columns if col not in dropped]
    result = run_variant(name, base, config, output, config["targets"]["horizons"],
                         variant=((42,), allow), workers=4)
    (output / name / "removed_columns.json").write_text(json.dumps(removed, indent=2))
    return result


def peak_rss_kib(pid):
    # Soma RSS de pai e descendentes em uma fotografia; memória compartilhada
    # pode ser contada em mais de um processo.
    listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = [list(map(int, line.split())) for line in listing.splitlines() if line.strip()]
    descendants = {pid}
    while True:
        expanded = descendants | {p for p, parent, _ in rows if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    return sum(rss for p, _, rss in rows if p in descendants)


def full_run(output, release, config_path):
    output.mkdir(parents=True)
    manifest = json.loads((release / "release.json").read_text())
    shutil.copy(release / "training_base.csv", output / "training_base.csv")
    digest = hashlib.sha256((output / "training_base.csv").read_bytes()).hexdigest()
    if digest != manifest["training_base_sha256"]:
        raise ValueError("Hash da base difere do manifesto da versão")
    shutil.copy(config_path, output / "config.yaml")
    outcomes = {}
    cohort = None
    for name in FEATURE_VARIANTS:
        before = os.times()
        with (output / f"{name}.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, __file__, str(output), "--worker", name],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            peak = 0
            while proc.poll() is None:
                peak = max(peak, peak_rss_kib(proc.pid))
                time.sleep(1)
        if proc.returncode:
            raise RuntimeError(f"Falha ao treinar {name}; consulte {output / (name + '.log')}")
        after = os.times()
        result = json.loads((output / name / "result.json").read_text())
        if cohort is not None and result["cohort_sha256"] != cohort:
            raise ValueError("Variantes em coortes diferentes")
        cohort = result["cohort_sha256"]
        outcomes[name] = {
            "feature_count": len(result["feature_allowlist"]), "train_seconds": result["train_seconds"],
            "cpu_seconds_including_evaluation": (after.children_user + after.children_system
                                                 - before.children_user - before.children_system),
            "peak_summed_rss_kib": peak, "model_bytes": result["model_bytes"],
            "metrics": result["evaluation"]["by_horizon"],
            "by_fold": result["evaluation"]["by_horizon_and_fold"],
        }
        print(f"{name}: {result['evaluation']['n_scored']} previsões", flush=True)

    # Alterar apenas a regra indicada; o avaliador reajusta o conformal com
    # desfechos maduros de folds anteriores quando essa camada está ligada.
    config = yaml.safe_load((output / "config.yaml").read_text())
    base = pd.read_csv(output / "training_base.csv", dtype={"SE": str})
    artifact = json.loads((release / "validation_predictions.json").read_text())
    deployed = json.loads((release / "walk_forward_evaluation.json").read_text())
    control_raw = json.loads((output / "all_features/validation_predictions.json").read_text())
    control_report = json.loads((output / "all_features/result.json").read_text())["evaluation"]
    if evaluation_cohort_hash(artifact, deployed, decimals=8) != evaluation_cohort_hash(
            control_raw, control_report, decimals=8):
        raise ValueError("A versão de produção foi pontuada em uma coorte diferente")
    layers = {}
    for name in ("production", "no_profile", "no_conformal", "no_gating", "no_level_bias"):
        cfg = deepcopy(config)
        if name == "no_profile":
            cfg["postprocessing"]["band_shape"]["enabled"] = False
        elif name == "no_conformal":
            cfg["quantile_conformal"]["enabled"] = False
        elif name == "no_gating":
            cfg["phantom_gating"]["enabled"] = False
        elif name == "no_level_bias":
            cfg["postprocessing"]["median_bias_factor"] = 1.0
        start = time.monotonic()
        report, _, _ = evaluate(artifact, base, cfg)
        metrics = {str(h): row for h, row in report["by_horizon"].items()}
        # Releases anteriores ao manifesto v2 pontuaram limites inclusivos
        # contra alvos com erro de ponto flutuante da transformação log1p.
        stable_metrics = ["mae", "wis_total", "width_90"]
        if manifest.get("schema_version", 1) >= 2:
            stable_metrics.append("coverage_90")
        if name == "production" and any(abs(row[k] - deployed["by_horizon"][h][k]) > 1e-9
                                        for h, row in metrics.items() for k in stable_metrics):
            raise RuntimeError("O replay diverge das métricas da versão de origem")
        for h, row in metrics.items():
            if row["n_scored"] != outcomes["all_features"]["metrics"][h]["n_scored"]:
                raise RuntimeError("Contagem divergente na comparação das camadas")
        layers[name] = {"evaluation_seconds": time.monotonic() - start, "metrics": metrics,
                        "by_fold": report["by_horizon_and_fold"]}
    summary = {"release": release.name, "base_sha256": digest, "feature_cohort_sha256": cohort,
               "feature_variants": outcomes, "layer_variants": layers}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", type=Path, default=ROOT / "models/current")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--worker", choices=FEATURE_VARIANTS, help=argparse.SUPPRESS)
    opts = parser.parse_args()
    output = opts.output.resolve()
    if not output.is_relative_to(ROOT / "logs"):
        parser.error("Diretório experimental deve ficar em logs/")
    if opts.worker:
        train_variant(output, opts.worker)
    else:
        full_run(output, opts.release.resolve(), opts.config.resolve())
