#!/usr/bin/env python3
"""Episense auto-update pipeline (cron).

1. Collect fresh InfoDengue data (API) and merge into the full history
   (recent weeks are revised/nowcast, so they replace by SE).
2. Collect fresh OpenMeteo daily data and aggregate to weekly epi weeks
   (Brazilian calendar, Sunday start).
3. Reprocess data/processed/episense_base.csv.
4. If the base changed (new/revised weeks), retrain all models.

The API (api/main.py) recarrega a base processada e os modelos por mtime no
próximo request, então as previsões usam os dados/modelos frescos sem restart.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

from scripts.retrain_pipeline import train_and_publish


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
        week = last_se(base)
        try:
            from config.config import load_config
            model_root = ROOT / load_config()["paths"]["models"]
            release = train_and_publish(model_root, LOGS, trained_week=week)
        except (OSError, RuntimeError) as exc:
            print(f"Retreino não publicado: {exc}")
            sys.exit(1)
        state.write_text(digest)
        print(f"Versão {release.name} publicada para SE {week}")
        sys.exit(0)

    print(f"base inalterada (ultima SE {last_se(base)}) - sem retreino")


if __name__ == "__main__":
    main()
