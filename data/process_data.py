"""
Data Processing and Feature Engineering Module for Episense
Merges dengue and weather data, creates epidemiological week format (SE = YYYYWW),
and prepares the base dataset for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))       # project root (data.epiweeks)
import yaml

from data.epiweeks import weeks_in_year, epiweek_to_date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisenseDataProcessor:
    """Processes and merges dengue and weather data for Episense modeling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_raw_dir = Path(self.config['paths']['data_raw'])
        self.data_processed_dir = Path(self.config['paths']['data_processed'])
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.geocode = self.config['project']['geocode']
        self.target_col = self.config['project']['target_variable']
        self.se_format = self.config['project']['se_format']  # YYYYWW
        
    def load_infodengue_data(self) -> pd.DataFrame:
        """Load the latest InfoDengue data.

        Prefers the accumulated full history (infodengue_cg_full.csv), which is
        refreshed by collect_infodengue.py with the latest revised weeks; falls
        back to the newest timestamped fetch otherwise.
        """
        full = self.data_raw_dir / "infodengue/infodengue_cg_full.csv"
        if full.exists():
            logger.info(f"Loading InfoDengue data from {full}")
            return pd.read_csv(full)
        files = list(self.data_raw_dir.glob("infodengue/infodengue_cg_*.csv"))
        if not files:
            logger.error("No InfoDengue data found. Run collect_infodengue.py first.")
            return pd.DataFrame()
        
        latest = max(files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading InfoDengue data from {latest}")
        df = pd.read_csv(latest)
        return df
    
    def load_openmeteo_data(self) -> pd.DataFrame:
        """Load the latest OpenMeteo weekly data (largest end_date, not mtime)."""
        from data.collect_openmeteo import latest_weekly_file
        latest = latest_weekly_file(self.data_raw_dir / "openmeteo")
        if latest is None:
            logger.error("No OpenMeteo data found. Run collect_openmeteo.py first.")
            return pd.DataFrame()
        
        logger.info(f"Loading OpenMeteo data from {latest}")
        df = pd.read_csv(latest)
        return df
    
    def merge_data(self, dengue_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge dengue and weather data on epidemiological week (SE)."""
        logger.info(f"Dengue data shape: {dengue_df.shape}")
        logger.info(f"Weather data shape: {weather_df.shape}")
        
        # Ensure SE is string with proper format
        dengue_df['SE'] = dengue_df['SE'].astype(str).str.zfill(6)
        weather_df['SE'] = weather_df['SE'].astype(str).str.zfill(6)
        
        # Merge on SE
        merged = pd.merge(dengue_df, weather_df, on='SE', how='left', suffixes=('', '_weather'))
        
        # Handle duplicate columns from merge
        cols_to_drop = [c for c in merged.columns if c.endswith('_weather') and c.replace('_weather', '') in merged.columns]
        merged = merged.drop(columns=cols_to_drop)
        
        logger.info(f"Merged data shape: {merged.shape}")
        logger.info(f"SE range: {merged['SE'].min()} to {merged['SE'].max()}")

        # ANCHOR GUARD: the model must see climate ONLY from the last InfoDengue
        # week. Compute each week's climate days from the RAW weather (before
        # ffill), then, after ensuring continuous weeks, drop any trailing week
        # whose climate is missing or partial (<7 days). The prediction anchor
        # therefore only ever advances to a dengue week with COMPLETE climate of
        # its own week - never partial, never ffilled from an older week.
        clima_dias = None
        if 'week_start' in merged.columns and 'week_end' in merged.columns:
            dias = (pd.to_datetime(merged['week_end']) - pd.to_datetime(merged['week_start'])).dt.days + 1
            clima_dias = dict(zip(merged['SE'], dias))  # dict: robusto a SE duplicada

        # Check for missing weeks
        merged = self._ensure_continuous_weeks(merged)

        if clima_dias is not None:
            while len(merged) > 1:
                last_se = merged.iloc[-1]['SE']
                days = clima_dias.get(last_se, pd.NA)
                if pd.isna(days) or days < 7:
                    motivo = 'clima ausente' if pd.isna(days) else f'clima parcial ({int(days)}/7 dias)'
                    logger.warning(f"Anchor guard: SE {last_se} removida ({motivo}) - ancora recua para a semana anterior")
                    merged = merged.iloc[:-1]
                else:
                    break
        
        return merged
    
    def _ensure_continuous_weeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure continuous epidemiological weeks, filling missing with 0 cases.

        Uses the BRAZILIAN epidemiological calendar (weeks start on SUNDAY;
        week 1 contains January 4th; week 53 exists only in years that really
        have it - 2014, 2020, 2025 in this dataset). The week grid is derived
        from weeks_in_year(), so no artificial week is ever injected and no
        real week is dropped (ISO week numbers differ from the Brazilian ones).
        """
        df = df.copy()
        df = df.sort_values('SE').reset_index(drop=True)
        
        # Create complete range of epidemiological weeks
        min_se = int(df['SE'].min())
        max_se = int(df['SE'].max())
        
        all_weeks = []
        for year in range(min_se // 100, max_se // 100 + 1):
            for week in range(1, weeks_in_year(year) + 1):
                se = f"{year}{week:02d}"
                se_int = int(se)
                if min_se <= se_int <= max_se:
                    all_weeks.append(se)
        
        all_weeks_df = pd.DataFrame({'SE': all_weeks})
        all_weeks_df['SE'] = all_weeks_df['SE'].astype(str).str.zfill(6)
        
        # Merge to find missing weeks
        merged = pd.merge(all_weeks_df, df, on='SE', how='left')
        
        # Flag semanas imputadas (fabricadas) antes do fillna
        if 'casos' in merged.columns:
            merged['is_imputed'] = merged['casos'].isna()
            n_imputed = int(merged['is_imputed'].sum())
            if n_imputed > 0:
                logger.warning(f"Semanas imputadas (fabricadas): {n_imputed} semanas com casos ausente preenchidas com 0")
            merged['casos'] = merged['casos'].fillna(0).astype(int)
        else:
            merged['is_imputed'] = False
        
        # Forward fill weather variables (strict ffill - no future data used for imputation)
        weather_cols = [c for c in merged.columns if c not in ['SE', 'ano', 'semana', 'data_inicio_semana', 'casos']]
        for col in weather_cols:
            if merged[col].dtype in ['float64', 'int64']:
                merged[col] = merged[col].ffill()
        
        # Recreate ano and semana
        merged['ano'] = merged['SE'].str[:4].astype(int)
        merged['semana'] = merged['SE'].str[4:].astype(int)
        
        # Recreate date
        merged['data_inicio_semana'] = merged.apply(
            lambda row: self._epiweek_to_date(row['ano'], row['semana']), axis=1
        )
        
        logger.info(f"After ensuring continuous weeks: {merged.shape}")
        return merged
    
    def _epiweek_to_date(self, year: int, week: int) -> str:
        """Convert epidemiological year/week to date string (Brazilian calendar,
        week starts on SUNDAY - see data/epiweeks.py)."""
        return epiweek_to_date(year, week).strftime('%Y-%m-%d')
    
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from merged data."""
        df = df.copy()
        
        # NOTA: treino usa casos PURO (confirmados) sem substituicao por nowcast.
        # O nowcast (casos_est) so e usado na API/Dashboard nas ultimas 16 semanas (NOWCAST_WINDOW, ver api/main.py).
        # Log cases (target variable) - casos puro
        df['log_casos'] = np.log1p(df['casos'])
        
        # Seasonal features - usa weeks_in_year(ano) para anos com 53 semanas
        try:
            weeks = df['ano'].apply(lambda y: weeks_in_year(int(y)))
            df['sin_semana'] = np.sin(2 * np.pi * df['semana'] / weeks)
            df['cos_semana'] = np.cos(2 * np.pi * df['semana'] / weeks)
        except Exception:
            df['sin_semana'] = np.sin(2 * np.pi * df['semana'] / 52)
            df['cos_semana'] = np.cos(2 * np.pi * df['semana'] / 52)
        
        # Lag features for cases - FIXED: advanced convention shift N (no leakage)
        # Previously legacy shift 0 for lag1 caused target leakage; now shift 1..8
        df['log_lag1'] = df['log_casos'].shift(1)
        df['log_lag2'] = df['log_casos'].shift(2)
        df['log_lag3'] = df['log_casos'].shift(3)
        df['log_lag4'] = df['log_casos'].shift(4)
        df['log_lag5'] = df['log_casos'].shift(5)
        df['log_lag6'] = df['log_casos'].shift(6)
        df['log_lag7'] = df['log_casos'].shift(7)
        df['log_lag8'] = df['log_casos'].shift(8)
        
        # Weather lags - FIXED to true lag (shift N, not N-1)
        if 'temp_mean_mean' in df.columns:
            df['temp_lag2'] = df['temp_mean_mean'].shift(2)
            df['temp_lag4'] = df['temp_mean_mean'].shift(4)
        elif 'temp_mean' in df.columns:
            df['temp_lag2'] = df['temp_mean'].shift(2)
            df['temp_lag4'] = df['temp_mean'].shift(4)
        
        if 'precip_total' in df.columns:
            df['precip_lag2'] = df['precip_total'].shift(2)
            df['precip_lag4'] = df['precip_total'].shift(4)
        elif 'precip' in df.columns:
            df['precip_lag2'] = df['precip'].shift(2)
            df['precip_lag4'] = df['precip'].shift(4)
        
        if 'humidity_mean' in df.columns:
            df['humidity_lag2'] = df['humidity_mean'].shift(2)
            df['humidity_lag4'] = df['humidity_mean'].shift(4)
        
        # Rt / p_rt / nivel lags: REMOVED (Bug fix from code review).
        # Rt, p_rt and nivel from InfoDengue are nowcasts that are revised
        # retrospectively (the CSV carries the FINAL revised value for every
        # week). Using them as features - even with shift(2) - leaks future
        # information into the training/validation window because the final
        # value of week t-k incorporates notifications that arrived AFTER the
        # prediction date. They must NOT be used as features unless versioned
        # (as-published) nowcast data is available.
        # The raw columns (p_rt, Rt, nivel) remain in the CSV for reference but
        # are excluded from feature selection (see features.py select_features).
        
        return df
    
    def save_processed(self, df: pd.DataFrame, filename: str = "episense_base.csv") -> Path:
        """Save processed data."""
        output_path = self.data_processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path} ({df.shape})")
        return output_path
    
    def load_processed(self, filename: str = "episense_base.csv") -> pd.DataFrame:
        """Load processed data."""
        path = self.data_processed_dir / filename
        if not path.exists():
            logger.error(f"Processed file not found: {path}")
            return pd.DataFrame()
        logger.info(f"Loading processed data from {path}")
        return pd.read_csv(path)
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Run the complete data processing pipeline."""
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        dengue_df = self.load_infodengue_data()
        weather_df = self.load_openmeteo_data()
        
        if dengue_df.empty or weather_df.empty:
            logger.error("Missing raw data. Run collectors first.")
            return pd.DataFrame()
        
        # Merge
        merged = self.merge_data(dengue_df, weather_df)
        
        # Create base features
        featured = self.create_base_features(merged)
        
        # Save
        self.save_processed(featured)
        
        logger.info("Data processing pipeline complete!")
        return featured


def main():
    processor = EpisenseDataProcessor()
    df = processor.run_full_pipeline()
    print(f"Processed data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df[['SE', 'casos', 'log_casos', 'log_lag1', 'log_lag2', 'temp_lag2', 'temp_lag4']].head(10))
    print(df[['SE', 'casos', 'log_casos', 'log_lag1', 'log_lag2', 'temp_lag2', 'temp_lag4']].tail(10))


if __name__ == "__main__":
    main()