"""
Feature Engineering EXPERIMENTAL para Episense-Teste (melhoria de metricas).

Base: versao Aedex-Ultra (expand_extra) + NOVAS familias de features que atacam
a degradacao h5-h8 e a sazonalidade anual da dengue:

  expand_seasonal (flag):
    - log_casos_lag52 / log_casos_lag53  (mesma semana do ANO ANTERIOR)
    - log_casos_yoy52 = log_casos - log_casos_lag52  (crescimento anual, log)
    - casos_ytd (acumulado no ano epidemiologico ate a semana atual)
    - log_casos_ytd = log1p(casos_ytd)
    - casos_ytd_rate (media semanal do acumulado)
    - ytd_share_prev_year = casos_ytd / (casos_ytd_prev_year + eps) -> razao
      do acumulado atual vs o mesmo ponto do ano passado

  expand_weather_long (flag):
    - lags 7..12 de precip/temp/umid (efeito climatico de longo prazo em
      criadouros: chuva acumulada de 2-3 meses)
    - acumulados de precip em janelas 16/20/26 semanas
    - interacao temp x umid com medias moveis (estresse termico prolongado)

Convencao avancada: log_lagN = log_casos.shift(N), NUNCA shift(0).
Nenhuma feature usa informacao futura (todas shift >= 1 ou rolling com
min_periods=1 sobre o passado).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisenseFeatureEngineer:
    """Advanced feature engineering for dengue prediction (experimental)."""

    def __init__(self, config=None):
        self.config = config or {}
        self.feature_names = []
        self.target_horizons = [1, 2, 3, 4]
        self.expand_extra = bool(self.config.get('features', {}).get('expand_extra', False))
        self.expand_seasonal = bool(self.config.get('features', {}).get('expand_seasonal', False))
        self.expand_weather_long = bool(self.config.get('features', {}).get('expand_weather_long', False))

    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'log_casos',
                            max_lag: int = 12, convention: str = 'advanced') -> pd.DataFrame:
        """Create lag features for target variable (advanced: shift N)."""
        df = df.copy()
        prefix = target_col

        if convention == 'legacy':
            if target_col == 'log_casos':
                df['log_lag1'] = df[target_col].shift(0)
                for n in range(2, max_lag + 1):
                    df[f'log_lag{n}'] = df[target_col].shift(n - 1)
            else:
                for n in range(1, max_lag + 1):
                    df[f'{prefix}_lag{n}'] = df[target_col].shift(n)
        else:
            for n in range(1, max_lag + 1):
                df[f'{prefix}_lag{n}'] = df[target_col].shift(n)

        return df

    def create_weather_lags(self, df: pd.DataFrame, weather_cols: List[str],
                            lags: List[int] = [1, 2, 3, 4, 5, 6]) -> pd.DataFrame:
        df = df.copy()
        for col in weather_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'log_casos',
                                windows: List[int] = [3, 4, 8, 12],
                                weather_cols: List[str] = None) -> pd.DataFrame:
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window, min_periods=1).max()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window, min_periods=1).min()
            df[f'{target_col}_rolling_sum_{window}'] = df[target_col].rolling(window, min_periods=1).sum()

        if weather_cols:
            for col in weather_cols:
                if col not in df.columns:
                    continue
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window, min_periods=1).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window, min_periods=1).min()
                    df[f'{col}_rolling_sum_{window}'] = df[col].rolling(window, min_periods=1).sum()

        return df

    def create_seasonal_features(self, df: pd.DataFrame, semana_col: str = 'semana',
                                 ano_col: str = 'ano') -> pd.DataFrame:
        df = df.copy()
        df['sin_semana'] = np.sin(2 * np.pi * df[semana_col] / 52)
        df['cos_semana'] = np.cos(2 * np.pi * df[semana_col] / 52)
        df['mes_aprox'] = (df[semana_col] * 12 / 52).astype(int).clip(1, 12)
        df['sin_mes'] = np.sin(2 * np.pi * df['mes_aprox'] / 12)
        df['cos_mes'] = np.cos(2 * np.pi * df['mes_aprox'] / 12)
        # ano_raw: normalizado POR FOLD (treino) em train.py - nunca global
        df['ano_raw'] = df[ano_col]
        df['verao'] = df[semana_col].isin(list(range(1, 14)) + list(range(49, 53))).astype(int)
        df['outono'] = df[semana_col].between(14, 26).astype(int)
        df['inverno'] = df[semana_col].between(27, 39).astype(int)
        df['primavera'] = df[semana_col].between(40, 48).astype(int)
        for k in [1, 2, 3]:
            df[f'sin_{k}_semana'] = np.sin(2 * np.pi * k * df[semana_col] / 52)
            df[f'cos_{k}_semana'] = np.cos(2 * np.pi * k * df[semana_col] / 52)
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        temp_cols = [c for c in df.columns if 'temp' in c.lower() and 'lag' in c.lower()]
        hum_cols = [c for c in df.columns if 'humid' in c.lower() and 'lag' in c.lower()]
        for t in temp_cols[:3]:
            for h in hum_cols[:3]:
                if t in df.columns and h in df.columns:
                    df[f'{t}_x_{h}'] = df[t] * df[h]
        precip_cols = [c for c in df.columns if 'precip' in c.lower() and 'lag' in c.lower()]
        for t in temp_cols[:2]:
            for p in precip_cols[:2]:
                if t in df.columns and p in df.columns:
                    df[f'{t}_x_{p}'] = df[t] * df[p]
        case_lags = [c for c in df.columns if c.startswith('log_lag')]
        for cl in case_lags[:4]:
            for t in temp_cols[:2]:
                if cl in df.columns and t in df.columns:
                    df[f'{cl}_x_{t}'] = df[cl] * df[t]
        return df

    def create_horizon_specific_features(self, df: pd.DataFrame,
                                         horizons: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        df = df.copy()
        for h in horizons:
            df[f'log_lag_h{h}'] = df['log_casos'].shift(h)
            weather_cols = [c for c in df.columns if any(w in c.lower() for w in ['temp', 'precip', 'humid'])
                            and 'lag' not in c.lower() and 'rolling' not in c.lower()]
            for wc in weather_cols[:5]:
                df[f'{wc}_lag_h{h}'] = df[wc].shift(h)
        return df

    def create_difference_features(self, df: pd.DataFrame,
                                   target_col: str = 'log_casos',
                                   lags: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_diff_{lag}'] = df[target_col].diff(lag)
        weather_cols = [c for c in df.columns if any(w in c.lower() for w in ['temp', 'precip', 'humid'])
                        and 'lag' not in c.lower() and 'rolling' not in c.lower()]
        for wc in weather_cols[:5]:
            for lag in [1, 2]:
                df[f'{wc}_diff_{lag}'] = df[wc].diff(lag)
        return df

    def create_ewma_features(self, df: pd.DataFrame, target_col: str = 'log_casos',
                             spans: List[int] = [3, 4, 8, 12],
                             weather_cols: List[str] = None) -> pd.DataFrame:
        """EWMA (media e desvio) do alvo e, quando pedido, do clima."""
        df = df.copy()
        for span in spans:
            df[f'{target_col}_ewma_{span}'] = df[target_col].ewm(span=span, min_periods=1).mean()
            df[f'{target_col}_ewmstd_{span}'] = df[target_col].ewm(span=span, min_periods=1).std()

        if weather_cols:
            for col in weather_cols:
                if col not in df.columns:
                    continue
                for span in spans:
                    df[f'{col}_ewma_{span}'] = df[col].ewm(span=span, min_periods=1).mean()
                    df[f'{col}_ewmstd_{span}'] = df[col].ewm(span=span, min_periods=1).std()
        return df

    def create_accumulation_features(self, df: pd.DataFrame,
                                     target_col: str = 'casos',
                                     windows: List[int] = [4, 8, 12, 26],
                                     weather_cols: List[str] = None) -> pd.DataFrame:
        """Acumulados do alvo (casos) e, quando pedido, do clima."""
        df = df.copy()
        for window in windows:
            df[f'{target_col}_acc_{window}'] = df[target_col].rolling(window, min_periods=1).sum()
            df[f'{target_col}_acc_rate_{window}'] = df[f'{target_col}_acc_{window}'] / window

        if weather_cols:
            for col in weather_cols:
                if col not in df.columns:
                    continue
                for window in windows:
                    df[f'{col}_acc_{window}'] = df[col].rolling(window, min_periods=1).sum()
        return df

    # ------------------------------------------------------------------
    # NOVAS FAMILIAS (experimento de features)
    # ------------------------------------------------------------------
    def create_seasonal_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lags da MESMA SEMANA do ano anterior (T-52/T-53) + YoY + YTD.

        A dengue tem ciclo anual forte; o baseline sazonal (T-52) ja e usado
        no denominador do RMSSE, entao dar o padrao anual ao modelo ataca
        diretamente h5-h8 (onde a correlacao curta esgota e o modelo degrada).
        """
        df = df.copy()
        if 'log_casos' not in df.columns:
            return df

        # Mesma semana do ano anterior (52 e 53 cobrem anos de 53 semanas)
        df['log_casos_lag52'] = df['log_casos'].shift(52)
        df['log_casos_lag53'] = df['log_casos'].shift(53)

        # Crescimento ano-a-ano (diferenca de log = razao em casos)
        df['log_casos_yoy52'] = df['log_casos'] - df['log_casos_lag52']
        df['log_casos_yoy53'] = df['log_casos'] - df['log_casos_lag53']

        # Rolling da razao anual (suaviza ruido de 1 semana)
        df['log_casos_yoy52_roll4'] = df['log_casos_yoy52'].rolling(4, min_periods=1).mean()

        # Acumulado no ano epidemiologico (ate a semana atual) - so passado
        if 'casos' in df.columns and 'ano' in df.columns:
            df['casos_ytd'] = df.groupby('ano')['casos'].cumsum()
            df['log_casos_ytd'] = np.log1p(df['casos_ytd'])
            df['casos_ytd_rate'] = df['casos_ytd'] / df['semana'].clip(lower=1)

        # YTD do ano passado (mesmo ponto do ciclo) + razao
        if 'casos_ytd' in df.columns and 'ano' in df.columns:
            df['casos_ytd_prev_year'] = df.groupby('ano')['casos_ytd'].shift(52)
            df['ytd_share_prev_year'] = df['casos_ytd'] / (df['casos_ytd_prev_year'] + 1e-6)
            df['log_ytd_ratio'] = np.log1p(df['casos_ytd']) - np.log1p(df['casos_ytd_prev_year'].fillna(0))

        logger.info("expand_seasonal: lags T-52/53, YoY, YTD e razoes anuais adicionados")
        return df

    def create_weather_long_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clima com lag longo (7-12) + acumulados de chuva de 16-26 semanas.

        O efeito da chuva em criadouros persiste por 2-3 meses; lags curtos
        (1-6) nao capturam a memoria longa do clima.
        """
        df = df.copy()
        weather_cols = [c for c in df.columns if any(w in c.lower() for w in ['temp', 'precip', 'humid'])
                        and 'lag' not in c.lower() and 'rolling' not in c.lower()
                        and 'ewma' not in c.lower() and 'acc_' not in c.lower()
                        and 'diff' not in c.lower()]

        # Lags 7-12 (so clima principal para nao explodir o pool)
        main = [c for c in ['temp_mean_mean', 'precip_total', 'humidity_mean', 'temp_mean_max',
                            'temp_mean_min', 'precip_mean', 'humidity_max'] if c in df.columns]
        for col in main:
            for lag in range(7, 13):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Acumulados longos de chuva (16/20/26 semanas)
        for col in [c for c in ['precip_total', 'precip_mean'] if c in df.columns]:
            for w in [16, 20, 26]:
                df[f'{col}_acc_{w}'] = df[col].rolling(w, min_periods=1).sum()

        # Media movel longa de temp/umid (estresse termico prolongado)
        for col in [c for c in ['temp_mean_mean', 'humidity_mean'] if c in df.columns]:
            for w in [16, 26]:
                df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w, min_periods=1).mean()

        # Interacao temp x umid com media movel (condicoes prolongadas)
        if 'temp_mean_mean_rolling_mean_16' in df.columns and 'humidity_mean_rolling_mean_16' in df.columns:
            df['temp16_x_hum16'] = df['temp_mean_mean_rolling_mean_16'] * df['humidity_mean_rolling_mean_16']

        # Chuva acumulada ponderada (efeito defasado 4-8 semanas)
        if 'precip_total_lag4' in df.columns and 'precip_total_lag8' in df.columns:
            df['precip_weighted_4_8'] = (2 * df['precip_total_lag4'] + df['precip_total_lag8']) / 3

        logger.info("expand_weather_long: lags 7-12, acumulados 16-26s e interacoes longas adicionados")
        return df

    # ------------------------------------------------------------------
    def select_features(self, df: pd.DataFrame, target_col: str = 'target',
                        exclude_cols: List[str] = None, all_horizons: List[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()

        default_exclude = ['SE', 'ano', 'semana', 'data_inicio_semana', 'mes_aprox',
                           'verao', 'outono', 'inverno', 'primavera',
                           target_col, 'target',
                           # InfoDengue: colunas que vazam futuro ou sao IDs
                           'casos_est', 'casos_est_min', 'casos_est_max',
                           'p_rt1', 'p_inc100k', 'Localidade_id', 'nivel', 'id',
                           'tweet', 'Rt', 'pop', 'casprov', 'casprov_est', 'casprov_est_min',
                           'casprov_est_max', 'casconf', 'notif_accum_year',
                           'municipio_nome', 'versao_modelo', 'data_iniSE',
                           'week_start', 'week_end',
                           'receptivo', 'transmissao', 'nivel_inc',
                           'p_rt', 'Rt', 'nivel']
        # Exclui TODAS as colunas target_h* (vazamento entre horizontes)
        if all_horizons:
            default_exclude.extend([f'target_h{h}' for h in all_horizons])

        if exclude_cols:
            default_exclude.extend(exclude_cols)

        feature_cols = [c for c in df.columns if c not in default_exclude
                        and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        self.feature_names = feature_cols

        target_cols_to_keep = [target_col]
        if all_horizons:
            target_cols_to_keep.extend([f'target_h{h}' for h in all_horizons])

        return df[feature_cols + target_cols_to_keep], feature_cols

    def run_full_feature_engineering(self, df: pd.DataFrame,
                                     convention: str = 'advanced',
                                     target_horizons: List[int] = [1, 2, 3, 4],
                                     legacy_mode: bool = False) -> pd.DataFrame:
        """Run complete feature engineering pipeline (shared treino/inferencia)."""
        logger.info("Starting advanced feature engineering (EXP)...")
        df = df.copy()

        se_col = df['SE'].copy() if 'SE' in df.columns else None

        if 'log_casos' not in df.columns:
            if 'casos' in df.columns:
                df['log_casos'] = np.log1p(df['casos'])
            else:
                raise ValueError("Neither 'log_casos' nor 'casos' found in dataframe")

        for h in target_horizons:
            df[f'target_h{h}'] = df['log_casos'].shift(-h)
        df['target'] = df['target_h1']

        # CASE RATE (per 100k pop)
        if 'pop' in df.columns and df['pop'].notna().any():
            pop = df['pop'].iloc[0] if df['pop'].notna().any() else 906092
            df['casos_rate'] = df['casos'] / pop * 100000
            df['log_casos_rate'] = np.log1p(df['casos_rate'])

        # Lags avancados: log_lagN = shift(N) - NUNCA shift(0) para lag1
        df = self.create_lag_features(df, 'log_casos', max_lag=12, convention=convention)

        if 'log_casos_rate' in df.columns:
            df = self.create_lag_features(df, 'log_casos_rate', max_lag=8, convention=convention)

        # Lags ponderados (janela serial 2-4 da dengue)
        if all(f'log_casos_lag{l}' in df.columns for l in [2, 3, 4]):
            df['log_casos_weighted_lag24'] = (df['log_casos_lag2'] + 2*df['log_casos_lag3'] + df['log_casos_lag4']) / 4
        if 'log_casos_rate' in df.columns and all(f'log_casos_rate_lag{l}' in df.columns for l in [2, 3, 4]):
            df['log_casos_rate_weighted_lag24'] = (df['log_casos_rate_lag2'] + 2*df['log_casos_rate_lag3'] + df['log_casos_rate_lag4']) / 4

        weather_cols = [c for c in df.columns if any(w in c.lower() for w in ['temp', 'precip', 'humid', 'wind'])
                        and 'lag' not in c.lower() and 'rolling' not in c.lower()
                        and 'ewma' not in c.lower() and 'acc_' not in c.lower()
                        and 'diff' not in c.lower()]
        logger.info(f"Weather columns for lags: {weather_cols}")

        # Weather lags (dengue delay 2-4 semanas)
        df = self.create_weather_lags(df, weather_cols, lags=[1, 2, 3, 4, 5, 6])

        # Indice ponderado lag24 do clima
        for col in weather_cols:
            if all(f'{col}_lag{l}' in df.columns for l in [2, 3, 4]):
                df[f'{col}_weighted_lag24'] = (df[f'{col}_lag2'] + 2*df[f'{col}_lag3'] + df[f'{col}_lag4']) / 4

        # Rolling
        df = self.create_rolling_features(df, 'log_casos', windows=[3, 4, 8, 12, 26],
                                          weather_cols=weather_cols[:5])

        # Sazonalidade
        df = self.create_seasonal_features(df)

        # Interacoes
        df = self.create_interaction_features(df)

        # Horizonte-especificas
        df = self.create_horizon_specific_features(df, target_horizons)

        # Diferencas
        df = self.create_difference_features(df, 'log_casos', lags=[1, 2, 3, 4])

        # EWMA
        df = self.create_ewma_features(df, 'log_casos', spans=[3, 4, 8, 12])

        # Acumulados
        df = self.create_accumulation_features(df, 'casos', windows=[4, 8, 12, 26])

        # ============================================================
        # EXPANSAO 1 (expand_extra - herdada do Aedex-Ultra):
        # EWMA/acumulados do clima, crescimento lag4, suavizacao taxa 100k
        # ============================================================
        if self.expand_extra:
            main_weather = [c for c in ['temp_mean_mean', 'precip_total', 'humidity_mean'] if c in df.columns]
            df = self.create_ewma_features(df, 'log_casos', spans=[3, 4, 8, 12],
                                           weather_cols=main_weather)
            df = self.create_accumulation_features(df, 'casos', windows=[4, 8, 12, 26],
                                                   weather_cols=main_weather)
            df['log_casos_diff_4'] = df['log_casos'].diff(4)
            if 'log_casos_rate' in df.columns:
                for w in [4, 8]:
                    df[f'log_casos_rate_rolling_mean_{w}'] = df['log_casos_rate'].rolling(w, min_periods=1).mean()
            logger.info("expand_extra: features climaticas expandidas adicionadas (EWMA/acumulados/diff4/rate rolling)")

        # ============================================================
        # EXPANSAO 2 (expand_seasonal - NOVA): padrao anual T-52/53, YoY, YTD
        # ============================================================
        if self.expand_seasonal:
            df = self.create_seasonal_lag_features(df)

        # ============================================================
        # EXPANSAO 3 (expand_weather_long - NOVA): clima lag 7-12, chuva 16-26s
        # ============================================================
        if self.expand_weather_long:
            df = self.create_weather_long_features(df)

        if legacy_mode or convention == 'legacy':
            logger.warning("legacy_mode nao suportado no modo EXP (use convention='advanced')")

        if se_col is not None:
            df['SE'] = se_col

        df, feature_cols = self.select_features(df, 'target', all_horizons=target_horizons)

        if se_col is not None and 'SE' not in df.columns:
            df['SE'] = se_col

        logger.info(f"Feature engineering complete (EXP). Shape: {df.shape}, Features: {len(feature_cols)}")
        return df


def create_target_variables(df: pd.DataFrame, horizons: List[int] = [1, 2, 3, 4],
                           target_col: str = 'log_casos') -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f'target_h{h}'] = df[target_col].shift(-h)
    df['target'] = df['target_h1']
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.process_data import EpisenseDataProcessor

    processor = EpisenseDataProcessor()
    df = processor.run_full_pipeline()

    if not df.empty:
        for flags in [{'expand_extra': True},
                      {'expand_seasonal': True},
                      {'expand_weather_long': True},
                      {'expand_extra': True, 'expand_seasonal': True, 'expand_weather_long': True}]:
            fe = EpisenseFeatureEngineer({'features': flags})
            df_adv = fe.run_full_feature_engineering(df, convention='advanced')
            print(f"\nFlags {flags} -> shape {df_adv.shape}, features: {len(fe.feature_names)}")
