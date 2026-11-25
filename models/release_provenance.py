"""Arquiva implementações e versões usadas pelo treino antes de iniciar os workers."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import platform
from zipfile import ZipFile, ZIP_DEFLATED


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def capture_runtime(root, candidate):
    files = set()
    for package in ("models", "data", "config"):
        files.update((root / package).glob("*.py"))
    for script in ("train", "retrain_pipeline", "evaluate_walk_forward", "recompute_metrics"):
        files.add(root / "scripts" / (script + ".py"))
    files.add(root / "requirements.txt")
    hashes = {}
    with ZipFile(candidate / "training_source.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = str(path.relative_to(root))
            content = path.read_bytes()
            hashes[name] = sha256(content).hexdigest()
            archive.writestr(name, content)
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                            if d.metadata.get("Name")}}
    (candidate / "training_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True))
    return hashes


def verify_sources(root, hashes):
    for name, digest in hashes.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Código do treino mudou durante a execução em {name}")
