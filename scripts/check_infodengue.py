#!/usr/bin/env python3
"""Episense InfoDengue freshness probe (cron).

Cheap check (1 request) comparing the latest SE in the InfoDengue API against
the last SE in the base. Runs the full update pipeline ONLY when the API has a
newer week; stays silent otherwise (watchdog pattern)."""
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')  # silencia NotOpenSSLWarning do urllib3 (cron mudo)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from data.collect_infodengue import (archive_first_weekly_counts, collect_infodengue_data,
                                     process_infodengue_data)

def api_max_se() -> str:
    df = process_infodengue_data(collect_infodengue_data())
    if df.empty:
        raise RuntimeError("Coleta InfoDengue vazia")
    archive_first_weekly_counts(df)
    return str(df["SE"].max()).zfill(6)


def base_max_se() -> str:
    df = pd.read_csv(ROOT / "data/processed/episense_base.csv", usecols=["SE"])
    return str(df["SE"].iloc[-1]).zfill(6)


def week_has_ended(se: str) -> bool:
    """True se a semana epidemiologica (SE) ja terminou (fim da semana < hoje).

    FIX (review Bug 3): o gate anterior checava o ARQUIVO de clima mais novo,
    que cobre apenas ate ontem - a semana corrente nunca estava completa nele,
    entao o probe quase nunca disparava e, se o cron diario falhasse, o sistema
    congelava silenciosamente. O calendario e a fonte da verdade: a SE nova so
    dispara o pipeline quando a semana dela ja fechou.
    """
    from data.epiweeks import epiweek_to_date
    ano, sem = int(se[:4]), int(se[4:])
    week_end = epiweek_to_date(ano, sem) + pd.Timedelta(days=6)
    return week_end < pd.Timestamp.now().normalize()


def main():
    try:
        api_se = api_max_se()
        base_se = base_max_se()
    except Exception as e:
        print(f"ERRO: probe InfoDengue falhou ({e})")
        sys.exit(1)

    if api_se > base_se:
        if not week_has_ended(api_se):
            # A SE nova ainda esta em curso (InfoDengue publica estimativas da
            # semana corrente). Ficar silencioso e tentar no proximo tick -
            # o anchor guard impede a base de avancar com clima parcial.
            sys.exit(0)
        print(f"InfoDengue liberou SE {api_se} (base em {base_se}) -> executando pipeline")
        r = subprocess.run([sys.executable, str(ROOT / "scripts/update_pipeline.py")])
        sys.exit(r.returncode)
    sys.exit(0)  # silent: nothing new


if __name__ == "__main__":
    # Old host cron entries invoke this script without arguments.
    # Scheduled collection now belongs to each municipal dashboard.
    if sys.argv[1:] == ["--manual"]:
        main()
