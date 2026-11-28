"""Capture and replay the inputs used to publish a dashboard forecast."""
import gzip
import hashlib
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from data.epiweeks import add_epiweeks

QUANTILES = {0.05: "q05", 0.25: "q25", 0.5: "q50", 0.75: "q75", 0.95: "q95"}


def forecast_rows(origin, predictions):
    rows = []
    for h, quantiles in sorted(predictions.items()):
        row = {"h": h, "target_se": add_epiweeks(origin, h)}
        for tau, key in QUANTILES.items():
            values = quantiles.get(tau, ())
            cases = np.expm1(values[-1]) if len(values) else np.nan
            row[key] = int(round(float(cases))) if np.isfinite(cases) else None
        rows.append(row)
    return rows


def model_fingerprint(directory):
    digest = hashlib.sha256()
    paths = set(directory.glob("lgbm_*.pkl"))
    paths.update(directory.glob("quantile*.json"))
    paths.update(directory / name for name in ("feature_list.json", "normalization_params.json"))
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.name.encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def config_fingerprint(config):
    # Normalize numeric YAML keys to the strings used by the JSON issuance archive.
    normalized = json.loads(json.dumps(config, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inference_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = ("data/epiweeks.py", "data/features.py", "models/band_shape.py",
             "models/calibration.py", "models/inference.py")
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return digest.hexdigest()


def capture_inputs(folder, frame, engine):
    """Persist the exact input frame before inserting the archive record."""
    payload = gzip.compress(frame.to_csv(index=False).encode("utf-8"), mtime=0)
    digest = hashlib.sha256(payload).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}.csv.gz"
    if not target.exists():
        with NamedTemporaryFile(dir=folder, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                temporary.chmod(0o644)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    configured_root = Path(engine.config.get("paths", {}).get("models", "models")).resolve()
    directory = engine.model_dir.resolve()
    release = directory.name if directory.parent == configured_root / "releases" else None
    observations = frame["casos_observados"] if "casos_observados" in frame else frame["casos"]
    valid = observations.notna()
    if "is_imputed" in frame:
        valid &= ~frame["is_imputed"].fillna(False)
    observed = frame.loc[valid, "SE"]
    return {
        "input_sha256": digest,
        "release": release,
        "model_sha256": model_fingerprint(directory),
        "inference_sha256": inference_fingerprint(),
        "config": engine.config,
        "persistence": float(observations[valid].iloc[-1]) if valid.any() else None,
        "persistence_week": str(observed.iloc[-1]) if valid.any() else None,
    }


def load_inputs(folder, digest):
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Hash do snapshot inválido")
    raw = (folder / f"{digest}.csv.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Hash do snapshot diverge do registro")
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)), dtype={"SE": str},
                       float_precision="round_trip")


def official_forecast(archive, origin):
    """Return the archived scheduled issuance shared by API and dashboard."""
    import json

    if not archive.exists():
        return None
    entries = json.loads(archive.read_text(encoding="utf-8"))
    for entry in reversed(entries):
        if entry.get("origin_se") == origin and entry.get("issuance_id"):
            payload = {key: entry[key] for key in (
                "issuance_id", "origin_se", "generated_at", "forecast", "source"
            )}
            payload["model_release"] = (entry.get("audit") or {}).get("release")
            payload["persistence_shadow"] = entry.get("persistence_shadow", [])
            return payload
    return None
