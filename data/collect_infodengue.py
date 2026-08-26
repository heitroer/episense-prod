"""
Collect dengue data from InfoDengue API for Campo Grande, MS.
API: https://info.dengue.mat.br/api/alertcity
"""

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


def collect_infodengue_data() -> pd.DataFrame:
    """Collect dengue data from InfoDengue API."""
    logger.info(f"Fetching data from InfoDengue API: {API_URL}")
    
    try:
        response = requests.get(API_URL, timeout=60)
        response.raise_for_status()
        
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        logger.info(f"Downloaded {len(df)} rows from InfoDengue")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching InfoDengue data: {e}")
        return pd.DataFrame()


def process_infodengue_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw InfoDengue data into standard format.
    
    Keeps ONLY 'casos' (confirmed) and 'casos_est' (nowcast).
    All other case variants (casos_est_min, casos_est_max, casprov, casconf, etc.) are dropped.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Keep ONLY the required columns from InfoDengue
    keep_cols = ['SE', 'casos', 'casos_est']
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
    
    # Ensure SE is string with YYYYWW format
    df['SE'] = df['SE'].astype(str).str.zfill(6)
    
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
        df['casos'] = pd.to_numeric(df['casos'], errors='coerce').fillna(0).astype(int)
    
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
    df_new['SE'] = df_new['SE'].astype(str).str.zfill(6)

    if full_path.exists():
        df_full = pd.read_csv(full_path)
        df_full['SE'] = df_full['SE'].astype(str).str.zfill(6)
        df_full = df_full[~df_full['SE'].isin(df_new['SE'])]
        df_merged = pd.concat([df_full, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged = df_merged.sort_values('SE').reset_index(drop=True)
    return df_merged


def save_infodengue_data(df: pd.DataFrame) -> Path:
    """Save processed InfoDengue data with timestamp."""
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