"""
Collect dengue data from InfoDengue API for Campo Grande, MS.
API: https://info.dengue.mat.br/api/alertcity
"""

from __future__ import annotations
import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config
from data.epiweeks import epiweek_to_date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
DATA_RAW_DIR = Path(config['paths']['data_raw']) / "infodengue"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

GEOCODE = config['project']['geocode']  # 5002704 for Campo Grande
DISEASE = config['project']['disease']  # dengue

API_URL = f"https://info.dengue.mat.br/api/alertcity?geocode={GEOCODE}&disease={DISEASE}&format=csv&ew_format=SE"

def build_range_url(base_url: str, days: int | None = None) -> str:
    """Consulta desde o início do histórico local para incorporar revisões antigas."""
    from datetime import date, timedelta
    from data.epiweeks import date_to_epiweek
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo(config["data_sources"]["openmeteo"]["timezone"])).date()
    se_end = str(date_to_epiweek(today)).zfill(6)
    if days is None:
        history = DATA_RAW_DIR / "infodengue_cg_full.csv"
        se_start = str(int(pd.read_csv(history, usecols=["SE"])["SE"].min())) if history.exists() else "201001"
    else:
        se_start = str(date_to_epiweek(today - timedelta(days=days))).zfill(6)
    return (
        f"{base_url}"
        f"&ew_start={int(se_start[4:])}&ey_start={int(se_start[:4])}"
        f"&ew_end={int(se_end[4:])}&ey_end={int(se_end[:4])}"
    )


def collect_infodengue_data() -> pd.DataFrame:
    """Collect dengue data from InfoDengue API (janela de revisao CONSOLIDADA)."""
    url = build_range_url(API_URL)
    logger.info(f"Fetching data from InfoDengue API (range revisado): {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Parse CSV
        from io import BytesIO
        df = pd.read_csv(BytesIO(response.content), encoding="utf-8-sig")
        
        logger.info(f"Downloaded {len(df)} rows from InfoDengue")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching InfoDengue data: {e}")
        return pd.DataFrame()


def process_infodengue_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw InfoDengue data into standard format.
    
    Retain notifications, nowcasts and confirmed counts where available.
    The Dourados pipeline also consumes municipal covariates.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Keep ONLY the required columns from InfoDengue
    keep_cols = ['SE', 'casos', 'casos_est', 'casconf']
    if str(GEOCODE) != "5002704":
        city = config["project"]["location"].split(",")[0]
        if "municipio_nome" not in df or not df["municipio_nome"].str.strip().eq(city).all():
            raise ValueError(f"Resposta InfoDengue incompatível com {city}")
        keep_cols = list(df.columns)
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Standardize column names (identity mapping for the kept columns)
    col_mapping = {
        'SE': 'SE',
        'casos': 'casos',
        'casos_est': 'casos_est',
    }
    rename_dict = {k: v for k, v in col_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Ensure SE is string with YYYYWW format - robust to float \"202628.0\" from CSV numeric coercion
    def _norm_se(s):
        se_str = str(s).strip()
        if '.' in se_str:
            se_str = se_str.split('.')[0]
        return se_str.zfill(6)
    df['SE'] = df['SE'].apply(_norm_se)
    
    # Extract year and week
    df['ano'] = df['SE'].str[:4].astype(int)
    df['semana'] = df['SE'].str[4:].astype(int)
    
    # Convert SE to datetime (start of epidemiological week - BRAZILIAN calendar,
    # Sunday start, see data/epiweeks.py; NOT ISO/fromisocalendar)
    df['data_inicio_semana'] = df.apply(
        lambda row: epiweek_to_date(int(row['ano']), int(row['semana'])), axis=1
    )
    
    # Ensure casos is numeric
    if 'casos' in df.columns:
        df['casos'] = pd.to_numeric(df['casos'], errors='coerce').astype('Int64')
    
    # Sort by SE
    df = df.sort_values('SE').reset_index(drop=True)
    
    return df


def update_full_history(df_new: pd.DataFrame, full_path: Path) -> pd.DataFrame:
    """Merge fresh API rows into the accumulated full history (replace by SE).

    The InfoDengue /alertcity endpoint only returns the last few weeks, and
    recent weeks are REVISED (nowcast). Merging by SE keeps the full series
    while refreshing the latest weeks with their most current values.
    """
    df_new = df_new.copy()
    def _norm_se2(s):
        se_str = str(s).strip()
        if '.' in se_str:
            se_str = se_str.split('.')[0]
        return se_str.zfill(6)
    df_new['SE'] = df_new['SE'].apply(_norm_se2)

    if full_path.exists():
        df_full = pd.read_csv(full_path)
        df_full['SE'] = df_full['SE'].apply(_norm_se2)
        # Preserve campos históricos que esta coleta municipal não solicita.
        previous = df_full.set_index('SE')
        for col in df_full.columns.difference(df_new.columns):
            df_new[col] = df_new['SE'].map(previous[col])
        df_full = df_full[~df_full['SE'].isin(df_new['SE'])]
        df_merged = pd.concat([df_full, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged = df_merged.sort_values('SE').reset_index(drop=True)
    return df_merged


def archive_first_weekly_counts(df: pd.DataFrame) -> Path:
    """Preserve the first collection seen for each new latest API week."""
    from data.archive_weekly_cases import save_counts
    return save_counts(df[["SE", "casos"]].astype(object).fillna("").to_dict("records"), DATA_RAW_DIR / "first_seen")


def save_infodengue_data(df: pd.DataFrame) -> Path:
    """Save processed InfoDengue data with timestamp."""
    archive_first_weekly_counts(df)
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"infodengue_cg_{timestamp}.csv"
    filepath = DATA_RAW_DIR / filename
    
    df.to_csv(filepath, index=False)
    logger.info(f"Saved InfoDengue data to {filepath}")
    
    return filepath


def main():
    """Main collection function."""
    logger.info("Starting InfoDengue data collection...")
    
    # Collect data
    df_raw = collect_infodengue_data()
    
    if df_raw.empty:
        logger.error("No data collected. Exiting.")
        return
    
    # Process data
    df_processed = process_infodengue_data(df_raw)

    # Merge into the accumulated full history (refresh recent revised weeks,
    # append new SEs) and persist BOTH: the full file (used by the pipeline)
    # and the timestamped file (record of this fetch).
    full_path = DATA_RAW_DIR / "infodengue_cg_full.csv"
    df_full = update_full_history(df_processed, full_path)
    df_full.to_csv(full_path, index=False)
    logger.info(f"Full history updated: {len(df_full)} rows, SE {df_full['SE'].min()} to {df_full['SE'].max()}")

    # Save
    save_infodengue_data(df_processed)
    
    # Show summary
    print(f"\nData shape: {df_processed.shape}")
    print(f"Date range: {df_processed['SE'].min()} to {df_processed['SE'].max()}")
    print(f"Cases range: {df_processed['casos'].min()} to {df_processed['casos'].max()}")
    print(f"Total cases: {df_processed['casos'].sum()}")
    print(f"\nColumns: {df_processed.columns.tolist()}")
    print(f"\nSample data:")
    print(df_processed[['SE', 'casos', 'casos_est']].head(10))
    print(df_processed[['SE', 'casos', 'casos_est']].tail(10))


if __name__ == "__main__":
    main()