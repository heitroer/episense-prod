#!/usr/bin/env python3
"""Episense auto-update pipeline (cron).

1. Collect fresh InfoDengue data (API) and merge into the full history
   (recent weeks are revised/nowcast, so they replace by SE).
2. Collect fresh OpenMeteo daily data and aggregate to weekly epi weeks
   (Brazilian calendar, Sunday start).
3. Reprocess data/processed/episense_base.csv.
4. If the base changed (new/revised weeks), retrain all models.

The API (api/main.py) reloads the processed CSV by mtime on the next request,
so predictions always use the freshest data without a server restart.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)


def run(script: str, log: str) -> bool:
    r = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True)
    (LOGS / log).write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print(f"ERRO: {script} falhou (exit {r.returncode}) - veja logs/{log}")
        return False
    return True


def last_se(base: Path) -> str:
    import pandas as pd
    df = pd.read_csv(base, usecols=["SE"])
    return str(df["SE"].iloc[-1]).zfill(6)


def main():
    steps = [
        ("data/collect_infodengue.py", "update_infodengue.log"),
        ("data/collect_openmeteo.py", "update_openmeteo.log"),
        ("data/process_data.py", "update_process.log"),
    ]
    for script, log in steps:
        if not run(script, log):
            sys.exit(1)

    # Operational check: weather freshness vs dengue weeks. The merge is LEFT
    # on dengue with ffill for weather, so a lagging/partial OpenMeteo week is
    # silently carried over - surface it in the cron output instead.
    try:
        import pandas as pd
        from data.collect_openmeteo import latest_weekly_file
        base = pd.read_csv(ROOT / "data/processed/episense_base.csv", usecols=["SE"])
        wf = latest_weekly_file(ROOT / "data/raw/openmeteo")
        if wf is not None:
            w = pd.read_csv(wf, usecols=["SE", "week_start", "week_end"])
            w["SE"] = w["SE"].astype(str).str.zfill(6)
            last_base = str(base["SE"].iloc[-1]).zfill(6)
            last_w = str(w["SE"].max()).zfill(6)
            if last_w < last_base:
                print(f"AVISO: clima defasado (clima ate {last_w}, dengue ate {last_base}) - "
                      f"semanas finais usam ffill do clima anterior")
            else:
                row = w[w["SE"] == last_w].iloc[0]
                days = (pd.to_datetime(row["week_end"]) - pd.to_datetime(row["week_start"])).days + 1
                if days < 7:
                    print(f"AVISO: semana {last_w} do clima PARCIAL ({days}/7 dias) - "
                          f"valores serao corrigidos na proxima execucao")
    except Exception as e:
        print(f"AVISO: checagem de clima falhou ({e})")

    # Retrain only when the base actually changed (new/revised weeks)
    base = ROOT / "data/processed/episense_base.csv"
    digest = hashlib.md5(base.read_bytes()).hexdigest()
    state = LOGS / ".base_md5"
    changed = (not state.exists()) or state.read_text().strip() != digest

    if changed:
        # FIX (review Bug 2): the md5 state is written ONLY AFTER a successful
        # retrain; on failure the old state remains so the next tick retries.
        # Deployment gate (P7): backup dos artefatos estáveis antes do retreino;
        # se train falhar OU o gate reprovar (pior fold > budget vs baseline),
        # rollback dos modelos e NENHUMA promoção (sem md5 novo, sem restart).
        models_dir = ROOT / "models"
        backup = models_dir / "_backup_stable"
        backup_files = []
        try:
            if backup.exists():
                shutil.rmtree(backup)
            backup.mkdir(parents=True, exist_ok=True)
            keep_names = ("feature_list.json", "normalization_params.json",
                          "quantile_conformal.json", "deployment_gate.json",
                          "metrics_dataframe.csv", "validation_results.json",
                          "validation_predictions.json")
            for f in list(models_dir.glob("*.pkl")) + [models_dir / n for n in keep_names]:
                if f.exists():
                    shutil.copy2(f, backup / f.name)
                    backup_files.append(f.name)
            print(f"backup pre-retreino: {len(backup_files)} artefatos em models/_backup_stable")
        except Exception as e:
            print(f"AVISO: backup pre-retreino falhou ({e}) - rollback indisponível")

        def _restore_backup():
            if not backup_files:
                return False
            for f in backup.glob("*"):
                shutil.copy2(f, models_dir / f.name)
            return True

        ok = run("scripts/train.py", "update_retrain.log")
        if not ok:
            if _restore_backup():
                print("train FALHOU -> rollback aplicado (artefatos estáveis restaurados)")
            print(f"base atualizada (ultima SE {last_se(base)}) -> retreino FALHOU")
            sys.exit(1)

        gate = {}
        try:
            gp = models_dir / "deployment_gate.json"
            gate = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
        except Exception:
            gate = {}
        if not isinstance(gate, dict):
            gate = {}
        gate_active = bool(gate.get('enabled', False))
        gate_pass = bool(gate.get('pass', True))

        if gate_active and not gate_pass:
            thr = gate.get('threshold')
            print("GATE DE IMPLANTAÇÃO REPROVOU o retreino (pior fold > budget vs baseline):")
            for h, row in (gate.get('per_horizon') or {}).items():
                print(f"  h{h}: worst_fold_ratio={row.get('worst_fold_ratio')} (threshold {thr}) folds={row.get('folds')}")
            if _restore_backup():
                print(f"ROLLBACK aplicado: {len(backup_files)} artefatos restaurados de models/_backup_stable")
            else:
                print("AVISO: sem backup disponível - rollback impossível; VERIFICAR modelos em produção")
            print("Modelos antigos permanecem em produção; próxima SE com dados novos tentará novamente."
                  " Para promover manualmente: docker restart episense-prod-api (após revisar o gate).")
            sys.exit(1)

        # Métricas de validação com calibração conformal LOO (honesto): regenera
        # validation_results.json + metrics_dataframe.csv a partir das previsões novas
        # ANTES de promover (o dashboard exibe o estado final dos modelos + calibração).
        if not run("scripts/recompute_metrics.py", "update_recompute.log"):
            print("AVISO: recompute das métricas falhou - exibindo métricas do train (sem calibração LOO)")

        state.write_text(digest)
        # Restart the API so it loads the freshly retrained models.
        # Os modelos so sao carregados no startup do uvicorn (dados sao
        # relidos por mtime sem restart). Deploy atual = docker compose,
        # então reinicia o container da API (nome fixo no compose).
        # Usado antes: /Users/heitor/.hermes/.../episense-api-watchdog.sh
        # (apontava para Episense-Teste e uvicorn bare na porta 8001).
        rc = subprocess.run(
            ["docker", "restart", "episense-prod-api"],
            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"AVISO: restart da API falhou (exit {rc.returncode}): {rc.stderr.strip()}")
            print("Fallback manual: docker restart episense-prod-api")
        print(f"base atualizada (ultima SE {last_se(base)}) -> retreino OK"
              + (" (gate aprovado)" if gate_active else ""))
        sys.exit(0)

    print(f"base inalterada (ultima SE {last_se(base)}) - sem retreino")


if __name__ == "__main__":
    main()
