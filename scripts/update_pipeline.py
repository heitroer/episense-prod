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
import os
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
        ok = run("models/train.py", "update_retrain.log")
        if ok:
            state.write_text(digest)
            # Restart the API so it loads the freshly retrained models
            # (data freshness is handled by the mtime TTL in api/main.py).
            rc = subprocess.run(
                ["/Users/heitor/.hermes/scripts/episense-api-watchdog.sh", "restart"],
                capture_output=True, text=True)
            if rc.returncode != 0:
                print(f"AVISO: restart da API falhou (exit {rc.returncode}) - veja logs/api.log")
        print(f"base atualizada (ultima SE {last_se(base)}) -> retreino {'OK' if ok else 'FALHOU'}")
        sys.exit(0 if ok else 1)

    print(f"base inalterada (ultima SE {last_se(base)}) - sem retreino")


if __name__ == "__main__":
    main()
