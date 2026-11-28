"""Exercita rejeição, lock e paridade em diretórios temporários sobre uma versão real."""
import argparse
import copy
import fcntl
import json
import logging
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.config import load_config
from models.promotion import evaluate_promotion
from models.releases import active_models
from scripts.evaluate_walk_forward import evaluate
from scripts.retrain_pipeline import train_and_publish


def check(reference):
    reference = reference.resolve()
    cfg = load_config(str(ROOT / "config/config.yaml"))
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "current").symlink_to(reference, target_is_directory=True)
        with (root / ".retrain_lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                train_and_publish(root, root)
            except BlockingIOError:
                pass
            else:
                raise AssertionError("Treino concorrente admitido")
        with patch("scripts.retrain_pipeline.run_script", return_value=(False, "falha injetada")):
            try:
                train_and_publish(root, root)
            except RuntimeError as exc:
                assert "falha injetada" in str(exc)
            else:
                raise AssertionError("Falha de treino ignorada")
        assert active_models(root) == reference and not any((root / "releases").iterdir())

        candidate = root / "candidate"
        candidate.mkdir()
        shutil.copyfile(reference / "training_base.csv", candidate / "training_base.csv")
        base = pd.read_csv(candidate / "training_base.csv", dtype={"SE": str})
        history = {}
        report, _, _ = evaluate(json.loads((reference / "validation_predictions.json").read_text()),
                                base, cfg, history=history)
        (candidate / "validation_history.json").write_text(json.dumps(history))
        evaluation = candidate / "walk_forward_evaluation.json"
        evaluation.write_text(json.dumps(report))
        gate = evaluate_promotion(candidate, reference, cfg)
        assert gate["pass"], gate["failures"]
        strict = copy.deepcopy(cfg)
        strict["promotion_policy"]["max_ratio"]["wis_total"] = 0.5
        assert not evaluate_promotion(candidate, reference, strict)["pass"]
        report["by_horizon"][1]["coverage_90"] = 0
        evaluation.write_text(json.dumps(report))
        try:
            evaluate_promotion(candidate, reference, cfg)
        except ValueError as exc:
            assert "Paridade" in str(exc)
        else:
            raise AssertionError("Divergência do painel não detectada")
        assert active_models(root) == reference
        return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    result = check(args.reference)
    print(json.dumps({h: [row["n_paired"], row["n_peaks"]] for h, row in result["by_horizon"].items()}))
