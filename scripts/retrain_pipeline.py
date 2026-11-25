"""Promoção canônica de retreinos — compartilhada pelo dashboard e pelo cron.

O treino (in-process no dashboard ou via ``scripts/train.py`` no cron) escreve
os ``.pkl`` e o ``deployment_gate.json``. Este módulo centraliza o que precisa
acontecer **antes** e **depois** do treino para que os dois caminhos promovam
exatamente o mesmo estado:

* :func:`backup_models` — snapshot dos artefatos estáveis antes do treino;
* :func:`read_gate` / :func:`gate_passed` / :func:`gate_report` — leitura do
  deployment gate;
* :func:`restore_backup` — rollback em falha de treino ou reprovação do gate;
* :func:`recompute_canonical` — gera a avaliação com a cadeia publicada,
  a calibração conformal e as métricas legadas do dashboard;
* :func:`finalize_retrain` — orquestra gate + rollback + recompute.

O objetivo é impedir que o dashboard promova modelos sem backup, sem gate e
sem o pós-processamento canônico (P0 #2, análise de produção nov/2026).
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIRNAME = "_backup_stable"

# Artefatos que compõem o estado estável dos modelos. Inclui os .pkl (glob) e
# os metadados consumidos por inferência/métricas. ``quantile_mode.json`` e os
# arquivos de recalibração entram no rollback porque ``EpisenseInference`` os lê
# no startup — sem eles o rollback deixaria um estado misto.
BACKUP_KEEP_NAMES = (
    "feature_list.json",
    "normalization_params.json",
    "quantile_mode.json",
    "quantile_conformal.json",
    "quantile_recalibration.json",
    "quantile_recalibration_regime.json",
    "deployment_gate.json",
    "metrics_dataframe.csv",
    "validation_results.json",
    "validation_predictions.json",
    "fold_surto_summary.json",
    "walk_forward_evaluation.json",
)


def _artifact_paths(models_dir: Path) -> List[Path]:
    paths: List[Path] = list(models_dir.glob("*.pkl"))
    paths += [models_dir / name for name in BACKUP_KEEP_NAMES]
    return [p for p in paths if p.exists()]


def backup_models(models_dir: Path) -> Dict:
    """Copia pkls + metadados para ``models/_backup_stable`` antes do treino.

    Escreve primeiro em ``_backup_stable.tmp`` e só troca o backup anterior no
    final, para não deixar um snapshot parcial se o processo morrer no meio.
    Levanta exceção se não houver nenhum artefato ou se a cópia falhar.
    """
    models_dir = Path(models_dir)
    backup_dir = models_dir / BACKUP_DIRNAME
    tmp_dir = models_dir / f"{BACKUP_DIRNAME}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files: List[str] = []
    try:
        for src in _artifact_paths(models_dir):
            shutil.copy2(src, tmp_dir / src.name)
            files.append(src.name)
        if not files:
            raise RuntimeError(f"nenhum artefato encontrado em {models_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        tmp_dir.rename(backup_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return {"dir": backup_dir, "files": files}


def restore_backup(models_dir: Path, backup: Optional[Dict]) -> bool:
    """Restaura o snapshot em ``models_dir``.

    Usa ``shutil.copyfile`` (e não ``copy2``) de propósito: o rollback precisa
    atualizar o mtime dos artefatos restaurados, senão watchers de mtime (API e
    dashboard) continuariam servindo o engine carregado durante o retreino.
    Retorna ``False`` quando não há backup válido.
    """
    models_dir = Path(models_dir)
    backup = backup or {}
    backup_dir = Path(backup.get("dir") or (models_dir / BACKUP_DIRNAME))
    files = list(backup.get("files") or [])
    if not files or not backup_dir.exists():
        return False

    if any(not (backup_dir / name).is_file() for name in files):
        return False
    for name in files:
        shutil.copyfile(backup_dir / name, models_dir / name)
    for artifact in _artifact_paths(models_dir):
        if artifact.name not in files:
            artifact.unlink()
    return True


def read_gate(models_dir: Path) -> Dict:
    """Lê ``deployment_gate.json``; ausente/ilegível vira resultado inválido."""
    gate_path = Path(models_dir) / "deployment_gate.json"
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    except Exception as e:
        return {"_error": str(e)}
    return gate if isinstance(gate, dict) else {}


def gate_passed(gate: Optional[Dict]) -> bool:
    if not gate or type(gate.get("enabled")) is not bool or gate.get("pass") is not True:
        return False
    if not gate["enabled"]:
        return True
    horizons = gate.get("per_horizon")
    return isinstance(horizons, dict) and all(
        isinstance(horizons.get(str(h)), dict) and horizons[str(h)].get("folds")
        for h in range(1, 9)
    )


def gate_report(gate: Optional[Dict]) -> str:
    """Resumo legível do gate para logs."""
    gate = gate or {}
    threshold = gate.get("threshold")
    lines = [
        f"  h{h}: worst_fold_ratio={row.get('worst_fold_ratio')} "
        f"(threshold {threshold}) folds={row.get('folds')}"
        for h, row in (gate.get("per_horizon") or {}).items()
        if isinstance(row, dict)
    ]
    return "\n".join(lines) if lines else "  (sem detalhes por horizonte)"


def _tail(text: str, n_lines: int = 20) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n_lines:])


def run_script(
    script: str,
    args: Optional[List[str]] = None,
    *,
    timeout: int = 1800,
    log_path: Optional[Path] = None,
    env: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """Executa um script do projeto com o mesmo interpretador atual."""
    cmd = [sys.executable, str(ROOT / script)] + list(args or [])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired as e:
        output = f"TIMEOUT após {timeout}s ao executar {script}: {e}"
        ok = False

    if log_path is not None:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(output, encoding="utf-8")
        except OSError:
            pass

    return ok, output


def recompute_canonical(
    log_dir: Optional[Path] = None,
    *,
    conformal_log: str = "retrain_recompute_conformal.log",
    metrics_log: str = "retrain_recompute_metrics.log",
    timeout: int = 1800,
    env: Optional[Dict] = None,
) -> Dict:
    """Gera o walk-forward publicado e os indicadores legados do dashboard."""
    log_dir = Path(log_dir) if log_dir is not None else None

    ok_conformal, out_conformal = run_script(
        "scripts/evaluate_walk_forward.py",
        ["--write"],
        timeout=timeout,
        log_path=(log_dir / conformal_log) if log_dir else None,
        env=env,
    )
    ok_metrics, out_metrics = run_script(
        "scripts/recompute_metrics.py",
        timeout=timeout,
        log_path=(log_dir / metrics_log) if log_dir else None,
        env=env,
    )

    return {
        "conformal": ok_conformal,
        "conformal_tail": _tail(out_conformal),
        "metrics": ok_metrics,
        "metrics_tail": _tail(out_metrics),
    }


def finalize_retrain(
    models_dir: Path,
    backup: Optional[Dict],
    *,
    gate_mtime_floor: Optional[float] = None,
    log_dir: Optional[Path] = None,
    conformal_log: str = "retrain_recompute_conformal.log",
    metrics_log: str = "retrain_recompute_metrics.log",
    timeout: int = 1800,
) -> Dict:
    """Aplica gate + recompute canônico; faz rollback se o gate reprovar.

    ``gate_mtime_floor`` deve ser o timestamp capturado ANTES do treino. Ele
    garante que o ``deployment_gate.json`` lido foi regenerado por ESTE treino:
    se a geração do gate falhar, o arquivo antigo não pode aprovar modelos
    novos. 

    Retorna um dicionário com ``promoted``, ``gate``, ``gate_pass``,
    ``gate_fresh``, ``gate_reason``, ``rollback`` e ``recompute``. O chamador
    decide o que fazer com o resultado (marcador de SE treinada, caches etc.).
    """
    gate_path = Path(models_dir) / "deployment_gate.json"
    gate_fresh = True
    if gate_mtime_floor is not None:
        try:
            gate_fresh = (
                gate_path.exists()
                and gate_path.stat().st_mtime >= (float(gate_mtime_floor))
            )
        except OSError:
            gate_fresh = False

    gate = read_gate(models_dir)
    passed = gate_passed(gate) and gate_fresh
    if not gate_fresh:
        gate_reason = (
            "deployment_gate.json não foi regenerado por este treino "
            "(falha na geração do gate?)"
        )
    elif not gate_passed(gate):
        gate_reason = "gate inválido, incompleto ou reprovado"
    else:
        gate_reason = ""

    result = {
        "promoted": False,
        "gate": gate,
        "gate_pass": passed,
        "gate_fresh": gate_fresh,
        "gate_reason": gate_reason,
        "rollback": False,
        "recompute": {},
    }

    if not passed:
        result["rollback"] = restore_backup(models_dir, backup)
        return result

    try:
        result["recompute"] = recompute_canonical(
            log_dir=log_dir,
            conformal_log=conformal_log,
            metrics_log=metrics_log,
            timeout=timeout,
        )
    except Exception as exc:
        result["gate_reason"] = f"erro ao recalcular artefatos: {exc}"
        result["rollback"] = restore_backup(models_dir, backup)
        return result

    failed = [step for step in ("conformal", "metrics") if not result["recompute"].get(step)]
    if failed:
        result["gate_reason"] = "falha no recálculo de " + ", ".join(failed)
        result["rollback"] = restore_backup(models_dir, backup)
        return result
    result["promoted"] = True
    return result


def train_and_publish(root: Path, log_dir: Path, *, trained_week: Optional[str] = None) -> Path:
    """Treina e valida em um diretório privado antes de ativar a versão."""
    from config.config import load_config
    from models.inference import EpisenseInference
    from models.release_provenance import capture_runtime, utc_now, verify_sources
    from models.releases import active_models
    from models.promotion import evaluate_promotion

    root = root.resolve()
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    with (root / ".retrain_lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate = Path(tempfile.mkdtemp(prefix="model-", dir=releases))
        published = False
        try:
            started_at = utc_now()
            frozen_config = candidate / "training_config.yaml"
            shutil.copyfile(ROOT / "config/config.yaml", frozen_config)
            cfg = load_config(str(frozen_config))
            sources = capture_runtime(ROOT, candidate)
            snapshot = candidate / "training_base.csv"
            shutil.copyfile(ROOT / "data/processed/episense_base.csv", snapshot)
            with snapshot.open(newline="", encoding="utf-8") as handle:
                snapshot_week = max((row["SE"].zfill(6) for row in csv.DictReader(handle)), default="")
            if not snapshot_week or (trained_week and snapshot_week != trained_week):
                raise RuntimeError("A base mudou antes do treino. Repita com a semana atual.")
            env = dict(os.environ, EPISENSE_MODELS_DIR=str(candidate), EPISENSE_VP_DIR=str(candidate),
                       EPISENSE_TRAINING_BASE=str(snapshot), EPISENSE_TRAINING_CONFIG=str(frozen_config))
            for key in ("EPISENSE_QUICK", "EPISENSE_QUICK_HORIZONS"):
                env.pop(key, None)
            ok, output = run_script("scripts/train.py", timeout=7200,
                                    log_path=log_dir / "release_train.log", env=env)
            if not ok or not gate_passed(read_gate(candidate)):
                raise RuntimeError("Treino ou validação reprovada: " + _tail(output))
            result = recompute_canonical(log_dir, env=env)
            for step in ("conformal", "metrics"):
                if not result[step]:
                    raise RuntimeError(f"Falha no recálculo de {step}: {result[step + '_tail']}")
            engine = EpisenseInference(str(candidate), cfg)
            evaluation = json.loads((candidate / "walk_forward_evaluation.json").read_text())
            if evaluation.get("schema_version") != 2 or not evaluation.get("training_embargo_verified"):
                raise RuntimeError("Avaliação pareada do candidato ausente ou sem embargo verificável")
            for h in cfg["targets"]["horizons"]:
                score = evaluation["by_horizon"].get(str(h), {})
                if not score.get("n_scored") or score.get("wis_total") is None:
                    raise RuntimeError(f"Avaliação pareada sem resultados válidos em H{h}")
            for h in cfg["targets"]["horizons"]:
                for tau in cfg["quantiles"]:
                    loaded = engine.models.get(h, {}).get(tau, {})
                    if set(loaded) != set(cfg["model"]["ensemble"]["seeds"]):
                        raise RuntimeError(f"Modelos incompletos em h{h}, quantil {tau}")
            policy_report = evaluate_promotion(candidate, active_models(root), cfg)
            policy_json = json.dumps(policy_report, indent=2, allow_nan=False)
            (candidate / "promotion_gate.json").write_text(policy_json)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"promotion-{candidate.name}.json").write_text(policy_json)
            if not policy_report["pass"]:
                raise RuntimeError("Promoção reprovada: " + ", ".join(policy_report["failures"]))
            verify_sources(ROOT, sources)
            (candidate / ".retrain_last_se").write_text(snapshot_week)
            artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(candidate.iterdir()) if path.is_file()}
            (candidate / "release.json").write_text(json.dumps({
                "schema_version": 2, "started_at": started_at, "completed_at": utc_now(),
                "source_sha256": sources, "artifact_sha256": artifacts,
                "training_base_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "trained_week": snapshot_week,
            }, indent=2))
            # mkdtemp restringe leitura ao dono. O serviço da API roda com outro UID.
            candidate.chmod(0o755)
            pending = root / f".current-{candidate.name}"
            try:
                pending.symlink_to(candidate.relative_to(root), target_is_directory=True)
                pending.replace(root / "current")
                published = True
            finally:
                pending.unlink(missing_ok=True)
            return candidate
        finally:
            if not published:
                shutil.rmtree(candidate)
