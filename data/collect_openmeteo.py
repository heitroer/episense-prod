"""
Collect historical weather data from OpenMeteo Archive API for Campo Grande, MS.
"""

import requests
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime, timedelta
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config
from data.epiweeks import date_to_epiweek

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
DATA_RAW_DIR = Path(config['paths']['data_raw']) / "openmeteo"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

OM_CONFIG = config['data_sources']['openmeteo']
LATITUDE = OM_CONFIG['latitude']
LONGITUDE = OM_CONFIG['longitude']
TIMEZONE = OM_CONFIG['timezone']
BASE_URL = OM_CONFIG['base_url']

# Daily variables to collect
DAILY_VARS = OM_CONFIG['daily_variables']


def collect_openmeteo_data(start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """Collect historical weather data from OpenMeteo Archive API."""
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ','.join(DAILY_VARS),
        'timezone': TIMEZONE
    }
    
    logger.info(f"Fetching OpenMeteo data from {start_date} to {end_date}")
    logger.info(f"Variables: {DAILY_VARS}")
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' not in data:
            logger.error("No daily data in response")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['daily'])
        df['date'] = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
        
        # Rename columns to cleaner names
        rename_map = {
            'temperature_2m_max': 'temp_max',
            'temperature_2m_min': 'temp_min',
            'temperature_2m_mean': 'temp_mean',
            'precipitation_sum': 'precip',
            'relative_humidity_2m_mean': 'humidity',
            'wind_speed_10m_max': 'wind_max'
        }
        df = df.rename(columns=rename_map)
        
        logger.info(f"Downloaded {len(df)} daily records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching OpenMeteo data: {e}")
        return pd.DataFrame()


def aggregate_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather data to epidemiological weeks (SE = YYYYWW).

    Uses the BRAZILIAN epidemiological calendar (weeks start on SUNDAY) via
    data.epiweeks.date_to_epiweek - NOT isocalendar(), which numbers weeks
    with Monday start and would misalign weather with the dengue SEs.
    """
    df = df_daily.copy()
    
    # Add epidemiological week (Brazilian calendar, Sunday start)
    df['SE'] = df['date'].apply(date_to_epiweek)
    
    # Aggregate to weekly
    agg_dict = {
        'temp_max': ['max', 'mean', 'min'],
        'temp_min': ['max', 'mean', 'min'],
        'temp_mean': ['max', 'mean', 'min'],
        'precip': ['sum', 'mean', 'max'],
        'humidity': ['mean', 'min', 'max'],
        'wind_max': ['max', 'mean'],
        'date': ['min', 'max']
    }
    
    weekly = df.groupby('SE').agg(agg_dict).reset_index()
    
    # Flatten column names
    weekly.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in weekly.columns.values]
    
    # Rename for clarity
    rename_map = {
        'SE_': 'SE',
        'temp_max_max': 'temp_max_max',
        'temp_max_mean': 'temp_max_mean',
        'temp_max_min': 'temp_max_min',
        'temp_min_max': 'temp_min_max',
        'temp_min_mean': 'temp_min_mean',
        'temp_min_min': 'temp_min_min',
        'temp_mean_max': 'temp_mean_max',
        'temp_mean_mean': 'temp_mean_mean',
        'temp_mean_min': 'temp_mean_min',
        'precip_sum': 'precip_total',
        'precip_mean': 'precip_mean',
        'precip_max': 'precip_max_daily',
        'humidity_mean': 'humidity_mean',
        'humidity_min': 'humidity_min',
        'humidity_max': 'humidity_max',
        'wind_max_max': 'wind_max_max',
        'wind_max_mean': 'wind_max_mean',
        'date_min': 'week_start',
        'date_max': 'week_end'
    }
    weekly = weekly.rename(columns=rename_map)
    
    # Add year and week
    weekly['ano'] = weekly['SE'].str[:4].astype(int)
    weekly['semana'] = weekly['SE'].str[4:].astype(int)
    
    # Sort
    weekly = weekly.sort_values('SE').reset_index(drop=True)
    
    return weekly


def latest_weekly_file(raw_dir=None) -> Optional[Path]:
    """Weekly OpenMeteo file with the LARGEST end_date (parsed from the filename).

    Selection by mtime is unreliable: a failed/partial collection can write a
    file out of order. The filename carries the real data range
    (openmeteo_weekly_START_END_TIMESTAMP.csv).
    """
    import re
    base = Path(raw_dir) if raw_dir else DATA_RAW_DIR

    def end_date(f: Path) -> str:
        m = re.search(r"openmeteo_weekly_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})_", f.name)
        return m.group(1) if m else "0000-00-00"

    files = list(base.glob("openmeteo_weekly_*.csv"))
    return max(files, key=end_date) if files else None


def save_openmeteo_data(df_daily: pd.DataFrame, df_weekly: pd.DataFrame, start_date: str, end_date: str):
    """Save both daily and weekly data."""
    timestamp = datetime.now().strftime("%Y%m%d")
    
    daily_path = DATA_RAW_DIR / f"openmeteo_daily_{start_date}_{end_date}_{timestamp}.csv"
    weekly_path = DATA_RAW_DIR / f"openmeteo_weekly_{start_date}_{end_date}_{timestamp}.csv"
    
    df_daily.to_csv(daily_path, index=False)
    df_weekly.to_csv(weekly_path, index=False)
    
    logger.info(f"Saved daily data to {daily_path}")
    logger.info(f"Saved weekly data to {weekly_path}")


def main():
    """Main collection function."""
    logger.info("Starting OpenMeteo data collection...")
    
    # Collect historical data from 2010
    start_date = "2010-01-01"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    df_daily = collect_openmeteo_data(start_date, end_date)
    
    if df_daily.empty:
        logger.error("No data collected. Exiting.")
        return
    
    # Aggregate to weekly
    df_weekly = aggregate_to_weekly(df_daily)
    
    # Save
    save_openmeteo_data(df_daily, df_weekly, start_date, end_date)
    
    # Show summary
    print(f"\nDaily data shape: {df_daily.shape}")
    print(f"Weekly data shape: {df_weekly.shape}")
    print(f"Weekly SE range: {df_weekly['SE'].min()} to {df_weekly['SE'].max()}")
    print(f"\nWeekly columns: {df_weekly.columns.tolist()}")
    print(f"\nSample weekly data:")
    print(df_weekly[['SE', 'temp_mean_mean', 'precip_total', 'humidity_mean']].head(10))
    print(df_weekly[['SE', 'temp_mean_mean', 'precip_total', 'humidity_mean']].tail(10))


if __name__ == "__main__":
    main()