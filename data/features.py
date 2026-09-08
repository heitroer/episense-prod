"""
Feature Engineering EXPERIMENTAL para Episense-Prod (melhoria de metricas v2).

Base: versao Aedex-Ultra (expand_extra) + familias que atacam
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

  NOVO v2 (expand_climatology):
    - media climatologica 3 anos para casos e clima (lag52,104,156)
    - anomalias vs climatologia (casos, precip, temp, umid)
    - desvio padrao climatologico 3y e ratio

  NOVO v2 (expand_trend):
    - slopes lineares 4/8/12 semanas, momentum, CV, aceleracao
    - growth rates 1/2/4 semanas

  NOVO v2 (expand_outbreak):
    - semanas acima de limiares 50/100/200 em janelas 4/8/12
    - weeks_since_peak_52, intensidade surto, flags rising/falling

Convencao avancada: log_lagN = log_casos.shift(N), NUNCA shift(0).
Nenhuma feature usa informacao futura (todas shift >= 1 ou rolling com
min_periods sobre o passado, incluindo current que e conhecido em t).
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
    """Advanced feature engineering for dengue prediction (experimental v2)."""

    def __init__(self, config=None):
        self.config = config or {}
        self.feature_names = []
        self.target_horizons = [1, 2, 3, 4]
        self.expand_extra = bool(self.config.get('features', {}).get('expand_extra', False))
        self.expand_seasonal = bool(self.config.get('features', {}).get('expand_seasonal', False))
        self.expand_weather_long = bool(self.config.get('features', {}).get('expand_weather_long', False))
        self.expand_climatology = bool(self.config.get('features', {}).get('expand_climatology', False))
        self.expand_trend = bool(self.config.get('features', {}).get('expand_trend', False))
        self.expand_outbreak = bool(self.config.get('features', {}).get('expand_outbreak', False))
        self.expand_regime_conditional = bool(self.config.get('features', {}).get('expand_regime_conditional', False))
        self.expand_peak_timing = bool(self.config.get('features', {}).get('expand_peak_timing', False))
        self.outbreak_threshold = self.config.get('metrics', {}).get('classification_thresholds', {}).get('outbreak_threshold', 50)
        self.outbreak_thresholds = [self.outbreak_threshold, self.outbreak_threshold * 2, self.outbreak_threshold * 4]

    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'log_casos',
                            max_lag: int = 12, convention: str = 'advanced') -> pd.DataFrame:
        """Create lag features for target variable (advanced: shift N)."""
        df = df.copy()
        prefix = target_col

        if convention == 'legacy':
            raise ValueError("convention='legacy' desabilitado: shift 0 causava vazamento de dados (mesma semana). Use convention='advanced' (shift N)")

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
        # FIX: handle 53-week years (2014,2020,2025) - denominator per year, not fixed 52
        try:
            from data.epiweeks import weeks_in_year
            weeks = df[ano_col].apply(lambda y: weeks_in_year(int(y)))
        except Exception:
            weeks = 52
        df['sin_semana'] = np.sin(2 * np.pi * df[semana_col] / weeks)
        df['cos_semana'] = np.cos(2 * np.pi * df[semana_col] / weeks)
        df['mes_aprox'] = (df[semana_col] * 12 / weeks).astype(int).clip(1, 12)
        df['sin_mes'] = np.sin(2 * np.pi * df['mes_aprox'] / 12)
        df['cos_mes'] = np.cos(2 * np.pi * df['mes_aprox'] / 12)
        # ano_raw: normalizado POR FOLD (treino) em train.py - nunca global
        df['ano_raw'] = df[ano_col]
        df['verao'] = df[semana_col].isin(list(range(1, 14)) + list(range(49, 54))).astype(int)
        df['outono'] = df[semana_col].between(14, 26).astype(int)
        df['inverno'] = df[semana_col].between(27, 39).astype(int)
        df['primavera'] = df[semana_col].between(40, 48).astype(int)
        for k in [1, 2, 3]:
            df[f'sin_{k}_semana'] = np.sin(2 * np.pi * k * df[semana_col] / weeks)
            df[f'cos_{k}_semana'] = np.cos(2 * np.pi * k * df[semana_col] / weeks)
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
        # Interação biológica Temp × sazonalidade (EIP termodependente): mesma T em verão vs inverno tem competência vetorial distinta
        # Usa lag2 (janela extrínseca 8-12 dias) e sin/cos já existentes; respeita shift>=1
        try:
            temp_lag2 = None
            for c in ['temp_mean_mean_lag2', 'temp_mean_lag2', 'tempmed_lag2']:
                if c in df.columns:
                    temp_lag2 = c
                    break
            if temp_lag2 and 'sin_semana' in df.columns and 'cos_semana' in df.columns:
                df[f'{temp_lag2}_x_sin_semana'] = df[temp_lag2] * df['sin_semana']
                df[f'{temp_lag2}_x_cos_semana'] = df[temp_lag2] * df['cos_semana']
            # Variante com anomalia climatológica (desvio vs 3y) × sazonalidade
            if 'temp_mean_mean_anomaly_clim_lag1' in df.columns and 'sin_semana' in df.columns:
                df['temp_anomaly_clim_x_sin'] = df['temp_mean_mean_anomaly_clim_lag1'] * df['sin_semana'] if 'temp_mean_mean_anomaly_clim_lag1' in df.columns else df['temp_mean_mean_anomaly_clim'] * df['sin_semana']
            if 'precip_total_lag2' in df.columns and 'sin_semana' in df.columns:
                df['precip_lag2_x_sin_semana'] = df['precip_total_lag2'] * df['sin_semana']
        except Exception:
            pass
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

        # YTD do ano passado (mesmo ponto do ciclo) + razao - alinhado por semana (respeita 53 semanas)
        if 'casos_ytd' in df.columns:
            df['casos_ytd_prev_year'] = df.groupby('semana')['casos_ytd'].shift(1)
            df['ytd_share_prev_year'] = df['casos_ytd'] / (df['casos_ytd_prev_year'] + 1e-6)
            df['log_ytd_ratio'] = np.log1p(df['casos_ytd']) - np.log1p(df['casos_ytd_prev_year'].fillna(0))
            # razao YTD suavizada
            df['ytd_ratio_roll4'] = df['ytd_share_prev_year'].rolling(4, min_periods=1).mean()

        logger.info("expand_seasonal: lags T-52/53, YoY, YTD e razoes anuais adicionados (fix shift 52 global)")
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
    # NOVO v2: Climatologia 3 anos
    # ------------------------------------------------------------------
    def create_climatology_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Climatologia empirica 3 anos: media de lag52/104/156 e anomalias.

        Ataca diretamente o baseline sazonal usado no denominador do SPL/WIS:
        se o modelo sabe a expectativa climatologica para a mesma semana,
        pode calibrar melhor h5-h8 onde a autocorrelacao curta ja nao basta.
        Todas features usam apenas passado (shift >=52), sem vazamento.
        """
        df = df.copy()
        if 'log_casos' not in df.columns:
            return df

        # --- Casos: lags longos adicionais e climatologia 3y ---
        if 'log_casos_lag104' not in df.columns:
            df['log_casos_lag104'] = df['log_casos'].shift(104)
        if 'log_casos_lag156' not in df.columns:
            df['log_casos_lag156'] = df['log_casos'].shift(156)

        # Casos climatologia 3 anos (media e desvio das mesmas semanas em 1,2,3 anos atrás)
        clim_cols = [c for c in ['log_casos_lag52', 'log_casos_lag104', 'log_casos_lag156'] if c in df.columns]
        if clim_cols:
            df['log_casos_clim_mean_3y'] = df[clim_cols].mean(axis=1)
            df['log_casos_clim_std_3y'] = df[clim_cols].std(axis=1)
            df['log_casos_clim_median_3y'] = df[clim_cols].median(axis=1)
            # Anomalia vs climatologia (quanto a semana atual está acima do esperado)
            df['log_casos_anomaly_clim'] = df['log_casos'] - df['log_casos_clim_mean_3y']
            df['log_casos_anomaly_clim_lag1'] = df['log_casos'].shift(1) - df['log_casos_clim_mean_3y'].shift(1)
            # ratio casos vs clim (em escala log, mas ratio em casos)
            # evita divisao por zero: clip clim mean
            df['log_casos_clim_ratio'] = df['log_casos'] / df['log_casos_clim_mean_3y'].clip(lower=0.5)
            # desvio normalizado (z-score)
            df['log_casos_clim_zscore'] = (df['log_casos'] - df['log_casos_clim_mean_3y']) / (df['log_casos_clim_std_3y'] + 0.5)
            # YoY suavizado 3y
            df['log_casos_yoy_clim'] = df['log_casos'] - df['log_casos_clim_mean_3y']
            df['log_casos_yoy_clim_roll4'] = df['log_casos_yoy_clim'].rolling(4, min_periods=1).mean()

        # --- Clima climatologia 3y para precip/temp/humid (core) ---
        core_climate = [c for c in ['precip_total', 'temp_mean_mean', 'humidity_mean'] if c in df.columns]
        for col in core_climate:
            # lags 52,104,156 para clima
            for lag in [52, 104, 156]:
                lag_col = f'{col}_lag{lag}_clim'
                if lag_col not in df.columns:
                    df[lag_col] = df[col].shift(lag)
            lag_cols = [f'{col}_lag{lag}_clim' for lag in [52, 104, 156]]
            # media climatologica
            df[f'{col}_clim_mean_3y'] = df[lag_cols].mean(axis=1)
            df[f'{col}_clim_std_3y'] = df[lag_cols].std(axis=1)
            # anomalia atual vs clim
            df[f'{col}_anomaly_clim'] = df[col] - df[f'{col}_clim_mean_3y']
            # anomalia do lag1 vs clim daquela semana (feature sem vazamento para prever futuro)
            if f'{col}_lag1' in df.columns:
                # clim da semana correspondente ao lag1 (shift 1 da clim)
                df[f'{col}_anomaly_clim_lag1'] = df[f'{col}_lag1'] - df[f'{col}_clim_mean_3y'].shift(1)
            else:
                tmp_lag1 = df[col].shift(1)
                df[f'{col}_anomaly_clim_lag1'] = tmp_lag1 - df[f'{col}_clim_mean_3y'].shift(1)
            # desvio relativo
            df[f'{col}_clim_zscore'] = (df[col] - df[f'{col}_clim_mean_3y']) / (df[f'{col}_clim_std_3y'] + 1e-6)
            # precipitação: acumulada anomalia 4 semanas
            if 'precip' in col.lower():
                # anomalia acumulada 4 semanas
                df[f'{col}_anomaly_clim_acc4'] = df[f'{col}_anomaly_clim'].rolling(4, min_periods=1).sum()
                df[f'{col}_anomaly_clim_acc8'] = df[f'{col}_anomaly_clim'].rolling(8, min_periods=1).sum()

        # Temperatura/umid interação climatologica
        if 'temp_mean_mean_anomaly_clim' in df.columns and 'humidity_mean_anomaly_clim' in df.columns:
            df['temp_anom_x_hum_anom'] = df['temp_mean_mean_anomaly_clim'] * df['humidity_mean_anomaly_clim']
            df['temp_anom_x_precip_anom'] = df['temp_mean_mean_anomaly_clim'] * df.get('precip_total_anomaly_clim', 0)

        logger.info("expand_climatology: media 3y + anomalias vs climatologia (casos e clima) adicionados")
        return df

    # ------------------------------------------------------------------
    # NOVO v2: Tendencia / volatilidade
    # ------------------------------------------------------------------
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tendencia, momentum, volatilidade e aceleracao.

        Slopes lineares capturam a direcao da epidemia - crucial para h5-h8
        onde o modelo precisa extrapolar a curva (subida/descida) em vez de
        apenas copiar o ultimo valor. CV e aceleracao detectam inflexoes.
        """
        df = df.copy()
        if 'log_casos' not in df.columns:
            return df

        # Helper para slope via polyfit rolling
        def slope_for_window(series: pd.Series, window: int) -> pd.Series:
            # rolling com window exige min_periods=window para slope estável
            return series.rolling(window, min_periods=window).apply(
                lambda x: np.polyfit(np.arange(window), x, 1)[0] if len(x) == window else np.nan,
                raw=True
            )

        for window in [4, 8, 12]:
            # slope linear
            df[f'log_casos_slope_{window}'] = slope_for_window(df['log_casos'], window)
            # slope tambem com lag1 (tendencia ate semana passada, sem usar current para prever h>=1?)
            # mas log_casos atual é conhecido em t, então slope incluindo t é válido.
            # adiciona slope suavizado lag1
            df[f'log_casos_slope_{window}_lag1'] = df[f'log_casos_slope_{window}'].shift(1)

            # momentum: diferenca window
            df[f'log_casos_momentum_{window}'] = df['log_casos'] - df['log_casos'].shift(window)
            # volatility: CV = std/mean
            roll_mean = df['log_casos'].rolling(window, min_periods=1).mean()
            roll_std = df['log_casos'].rolling(window, min_periods=1).std()
            df[f'log_casos_cv_{window}'] = roll_std / (roll_mean.abs() + 0.5)
            # rolling std puro (ja tem mas adiciona CV)
            # burst: max - min
            roll_max = df['log_casos'].rolling(window, min_periods=1).max()
            roll_min = df['log_casos'].rolling(window, min_periods=1).min()
            df[f'log_casos_range_{window}'] = roll_max - roll_min

            # aceleracao: diferenca do momentum
            df[f'log_casos_accel_{window}'] = df[f'log_casos_momentum_{window}'].diff(1)
            # aceleracao do slope
            df[f'log_casos_slope_accel_{window}'] = df[f'log_casos_slope_{window}'].diff(1)

        # Aceleração curta (1-2 semanas) detecta inflexão rápida
        df['log_casos_accel_1_2'] = df['log_casos'].diff(1) - df['log_casos'].diff(1).shift(1)
        # growth rates em casos (nao log) para interpretabilidade 1/2/4 semanas
        for lag in [1, 2, 4]:
            # diff log já existe, mas adiciona crescimento relativo em casos
            shifted = df['casos'].shift(lag) if 'casos' in df.columns else np.expm1(df['log_casos'].shift(lag))
            current = df['casos'] if 'casos' in df.columns else np.expm1(df['log_casos'])
            df[f'casos_growth_rate_{lag}'] = (current - shifted) / (shifted + 5)
            # log growth ratio
            df[f'log_casos_growth_ratio_{lag}'] = df['log_casos'] / (df['log_casos'].shift(lag).clip(lower=0.5))

        # Tendencia relativa à climatologia (se disponivel)
        if 'log_casos_clim_mean_3y' in df.columns:
            df['trend_vs_clim_4'] = df['log_casos_momentum_4'] - (df['log_casos_clim_mean_3y'] - df['log_casos_clim_mean_3y'].shift(4))
            df['trend_vs_clim_8'] = df['log_casos_momentum_8'] - (df['log_casos_clim_mean_3y'] - df['log_casos_clim_mean_3y'].shift(8))

        # EWMA ratio: atual vs EWMA (desvio de tendencia suavizada)
        for span in [4, 8]:
            ewma_col = f'log_casos_ewma_{span}'
            if ewma_col in df.columns:
                df[f'log_casos_vs_ewma_{span}'] = df['log_casos'] - df[ewma_col]
                df[f'log_casos_vs_ewma_{span}_lag1'] = df[f'log_casos_vs_ewma_{span}'].shift(1)

        logger.info("expand_trend: slopes 4/8/12, momentum, CV, range, accel e growth rates adicionados")
        return df

    # ------------------------------------------------------------------
    # NOVO v2: Surto / fase epidemica
    # ------------------------------------------------------------------
    def create_outbreak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fase epidemica, contagem de semanas acima de limiar e tempo desde pico.

        Surto em dengue é auto-reforçado: semanas consecutivas acima de
        limiar + tempo desde ultimo pico informam se a epidemia está em
        ascensao, pico ou declínio - sinal forte para h5-h8.
        """
        df = df.copy()
        if 'casos' not in df.columns or 'log_casos' not in df.columns:
            return df

        # Contagem de semanas acima de limiares em janelas recentes (shift 1 para não usar current futuro)
        for thresh in self.outbreak_thresholds:
            indicator = (df['casos'] > thresh).astype(int)
            shifted = indicator.shift(1)
            for w in [4, 8, 12]:
                df[f'weeks_above_{thresh}_last{w}'] = shifted.rolling(w, min_periods=1).sum()
                df[f'weeks_above_{thresh}_last{w}_rate'] = df[f'weeks_above_{thresh}_last{w}'] / w
            # streak consecutivo atual (quantas semanas seguidas acima do limiar até t-1)
            # calculado via loop simples (n=816, trivial)
            streak = np.zeros(len(df), dtype=float)
            c = 0
            for i, val in enumerate(shifted.fillna(0).astype(int).values):
                if val == 1:
                    c += 1
                else:
                    c = 0
                streak[i] = c
            df[f'streak_above_{thresh}'] = streak

        # Tempo desde ultimo pico em 52 semanas (excluindo current)
        logc = df['log_casos'].values
        weeks_since_peak = np.full(len(df), np.nan)
        peak_value_52 = np.full(len(df), np.nan)
        for i in range(len(df)):
            start = max(0, i - 52)
            window = logc[start:i]  # exclui i
            if len(window) == 0:
                continue
            max_idx = int(np.argmax(window))
            peak_pos = start + max_idx
            weeks_since_peak[i] = i - peak_pos - 1
            peak_value_52[i] = window[max_idx]
        df['weeks_since_peak_52'] = weeks_since_peak
        df['weeks_since_peak_52_norm'] = weeks_since_peak / 52.0
        df['peak_value_52'] = peak_value_52
        df['dist_from_peak_52'] = df['log_casos'].shift(1) - df['peak_value_52']
        # tempo desde pico suavizado: se pico recente (<8 semanas), epidemia em declínio
        df['recent_peak_flag'] = (df['weeks_since_peak_52'] < 8).astype(int)
        df['old_peak_flag'] = (df['weeks_since_peak_52'] > 20).astype(int)

        # Intensidade de surto: soma ultimas 4 semanas vs baseline climatologico
        casos_sum4 = df['casos'].shift(1).rolling(4, min_periods=1).sum()
        # baseline: media 3y climatologica convertida para casos ou media rolling 52
        if 'log_casos_clim_mean_3y' in df.columns:
            clim_cases = np.expm1(df['log_casos_clim_mean_3y'].shift(1))
            df['outbreak_intensity_4'] = casos_sum4 / (clim_cases + 20)
            df['outbreak_intensity_8'] = df['casos'].shift(1).rolling(8, min_periods=1).sum() / (clim_cases.rolling(8, min_periods=1).sum() + 20)
        else:
            base = df['casos'].shift(1).rolling(52, min_periods=12).mean()
            df['outbreak_intensity_4'] = casos_sum4 / (base + 20)

        # Flags de fase epidemica simples
        # rising se diff positiva 2 semanas seguidas, falling idem
        diff1 = df['log_casos'].diff(1)
        df['epi_rising_2w'] = ((diff1 > 0) & (diff1.shift(1) > 0)).astype(int)
        df['epi_falling_2w'] = ((diff1 < 0) & (diff1.shift(1) < 0)).astype(int)
        df['epi_rising_3w'] = ((diff1 > 0) & (diff1.shift(1) > 0) & (diff1.shift(2) > 0)).astype(int)
        df['epi_falling_3w'] = ((diff1 < 0) & (diff1.shift(1) < 0) & (diff1.shift(2) < 0)).astype(int)

        # Acelerando: diff atual > media dos ultimos 8 diffs + 0.5*std
        if 'log_casos_diff_1' in df.columns:
            roll_mean_diff = df['log_casos_diff_1'].shift(1).rolling(8, min_periods=3).mean()
            roll_std_diff = df['log_casos_diff_1'].shift(1).rolling(8, min_periods=3).std()
            df['epi_accelerating'] = (df['log_casos_diff_1'] > roll_mean_diff + 0.5 * roll_std_diff.fillna(0)).astype(int)
            df['epi_decelerating'] = (df['log_casos_diff_1'] < roll_mean_diff - 0.5 * roll_std_diff.fillna(0)).astype(int)
        else:
            df['epi_accelerating'] = 0
            df['epi_decelerating'] = 0

        # Razao pico atual vs pico historico 52w (proximidade de recorde)
        df['peak_ratio_52'] = np.expm1(df['log_casos'].shift(1)) / (np.expm1(df['peak_value_52']) + 10)

        # Semanas acima de mediana climatologia
        if 'log_casos_clim_median_3y' in df.columns:
            above_clim = (df['log_casos'] > df['log_casos_clim_median_3y']).astype(int).shift(1)
            df['weeks_above_clim_last8'] = above_clim.rolling(8, min_periods=1).sum()

        logger.info("expand_outbreak: semanas acima limiar, weeks_since_peak, intensidade e flags de fase adicionados")
        return df

    # ------------------------------------------------------------------
    # NOVO v3: Regime condicional (lag sazonal gating)
    # ------------------------------------------------------------------
    def create_regime_conditional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lags sazonais condicionais ao regime do ano anterior + gating anti-phantom.

        Problema Fold2: q95 phantom S52 disparou porque lag52/53 puro + climatologia
        associam padrao seco de outono a surto, sem verificar se ano anterior teve
        surto na mesma semana. Gate condicional suprime falso positivo quando
        regime anterior era calma.

        Features:
        - prev_year_was_outbreak: 1 se lag52 casos > threshold (ano anterior teve surto na mesma semana)
        - log_casos_lag52_conditional = lag52 * flag  (só vale se veio de surto)
        - log_casos_lag53_conditional idem
        - yoy_gated = yoy52 * flag (crescimento só relevante se baseline foi surto)
        - clim_anomaly_gated: anomalia vs clim só quando clim indica outbreak
        - is_calm_season: flag mes calm (jun-set) para modelo aprender q95 calm não deve ser alto
        - precip_anomaly_x_regime: interação clima*regime
        """
        df = df.copy()
        if 'log_casos' not in df.columns:
            return df
        # flag se mesma semana do ano anterior teve surto (>50)
        # usa casos (não log) para threshold operacional
        casos = df['casos'] if 'casos' in df.columns else np.expm1(df['log_casos'])
        # lag52 casos do ano anterior
        if 'log_casos_lag52' in df.columns:
            casos_lag52 = np.expm1(df['log_casos_lag52'])
            prev_outbreak_52 = (casos_lag52 > self.outbreak_threshold).astype(int)
            # fillna 0 para primeiros 52 sem histórico
            prev_outbreak_52 = pd.Series(prev_outbreak_52).fillna(0).astype(int).values
            df['prev_year_was_outbreak_52'] = prev_outbreak_52
            # lags condicionais
            df['log_casos_lag52_conditional'] = df['log_casos_lag52'] * df['prev_year_was_outbreak_52']
            df['log_casos_lag52_calm'] = df['log_casos_lag52'] * (1 - df['prev_year_was_outbreak_52'])
            # yoy gated
            if 'log_casos_yoy52' in df.columns:
                df['log_casos_yoy52_gated'] = df['log_casos_yoy52'] * df['prev_year_was_outbreak_52']
                df['log_casos_yoy52_calm'] = df['log_casos_yoy52'] * (1 - df['prev_year_was_outbreak_52'])
        if 'log_casos_lag53' in df.columns:
            casos_lag53 = np.expm1(df['log_casos_lag53'])
            prev_outbreak_53 = (casos_lag53 > self.outbreak_threshold).astype(int)
            prev_outbreak_53 = pd.Series(prev_outbreak_53).fillna(0).astype(int).values
            df['prev_year_was_outbreak_53'] = prev_outbreak_53
            df['log_casos_lag53_conditional'] = df['log_casos_lag53'] * df['prev_year_was_outbreak_53']

        # flag estaçao calma vs outbreak (baseado na semana atual, conhecida em t)
        # semana já é conhecida, então is_calm é feature válida (não vazamento futuro)
        if 'semana' in df.columns:
            # tenta usar mes aproximado ou semana->mes; semanas 22-39 ~= jun-set (calm)
            # usa definicao config: calm_months 6,7,8,9 => semanas ~22-39
            # heuristica semana->mes: mes_aprox já existe, mas para simplicidade usa semana
            # 22=~jun inicio, 39=~set fim
            is_calm_week = df['semana'].between(22, 39).astype(int)
            df['is_calm_season'] = is_calm_week
            df['is_outbreak_season'] = 1 - is_calm_week
            # interação lag sazonal x estacao
            if 'log_casos_lag52' in df.columns:
                df['lag52_x_is_calm'] = df['log_casos_lag52'] * df['is_calm_season']
                df['lag52_x_is_outbreak'] = df['log_casos_lag52'] * df['is_outbreak_season']

        # anomalia climatologica gated: só relevante se anomalia positiva em estação outbreak
        if 'log_casos_anomaly_clim' in df.columns and 'is_outbreak_season' in df.columns:
            df['anomaly_clim_x_outbreak_season'] = df['log_casos_anomaly_clim'] * df['is_outbreak_season']
            df['anomaly_clim_x_calm_season'] = df['log_casos_anomaly_clim'] * df['is_calm_season']

        # clima x regime (precip anomalia só importa em estacao outbreak)
        if 'precip_total_anomaly_clim' in df.columns and 'is_outbreak_season' in df.columns:
            df['precip_anom_x_outbreak'] = df['precip_total_anomaly_clim'] * df['is_outbreak_season']
            df['precip_anom_x_calm'] = df['precip_total_anomaly_clim'] * df['is_calm_season']
        if 'temp_mean_mean_anomaly_clim' in df.columns and 'is_outbreak_season' in df.columns:
            df['temp_anom_x_outbreak'] = df['temp_mean_mean_anomaly_clim'] * df['is_outbreak_season']

        logger.info("expand_regime_conditional: lags condicionais + gating anti-phantom adicionados")
        return df

    # ------------------------------------------------------------------
    # NOVO v3: Peak timing (corrige S6 vs S15)
    # ------------------------------------------------------------------
    def create_peak_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de timing de pico para corrigir viés S6 vs S15.

        Histórico: pico mediano S07, mas 2022 S18 e 2023 S15 são tardios.
        Modelo puro sazonal prevê pico cedo (S06) porque prior puxa para média.
        Features de timing dão ao modelo sinal de que pico pode atrasar.

        - dist_to_typical_peak: distância circular (semanas) da semana atual ao pico típico (S07)
        - weeks_to_typical_peak: semanas até S07 (negativo se passou)
        - slope_x_dist: interação momentum atual x distância ao pico (se subindo longe do pico => picos tardio provável)
        - lag52_x_slope: gating sazonal (lag52 só vale se slope atual concorda com yoy)
        - yoy_x_slope: se yoy positivo e slope positivo => surto em ascensão tardia
        - trended_peak_lag: lag52 ajustado por tendência (lag52 + momentum*weeks_to_peak)
        """
        df = df.copy()
        if 'semana' not in df.columns or 'log_casos' not in df.columns:
            return df

        # pico típico histórico S07 (mediana dos picos 2015-2024)
        typical_peak_week = 7

        # distância circular mínima (considera ano circular 52 semanas)
        def circ_dist(w, peak=typical_peak_week, total=52):
            d = np.abs(w - peak)
            return np.minimum(d, total - d)

        df['dist_to_typical_peak'] = circ_dist(df['semana'])
        # semanas até pico (com sinal): positivo se antes do pico, negativo se depois
        # assume ano epi semanas 1-52/53, precisa lidar com passagem de ano
        # simplifica: se semana <=30, diff = peak - semana, senão peak+52-semana
        def weeks_to_peak(w, peak=typical_peak_week):
            # se w <= 30, pico no mesmo ano (peak - w), senão pico do próximo ano (peak+52 - w)
            return np.where(w <= 30, peak - w, peak + 52 - w)

        df['weeks_to_typical_peak'] = weeks_to_peak(df['semana'])
        df['is_before_typical_peak'] = (df['weeks_to_typical_peak'] > 0).astype(int)
        df['is_after_typical_peak'] = (df['weeks_to_typical_peak'] <= 0).astype(int)

        # interação slope x distância
        if 'log_casos_slope_4' in df.columns:
            df['slope4_x_dist_peak'] = df['log_casos_slope_4'] * df['dist_to_typical_peak']
            df['slope4_x_weeks_to_peak'] = df['log_casos_slope_4'] * df['weeks_to_typical_peak']
            # se está longe do pico mas subindo => pico tardio provável
            df['rising_far_from_peak'] = ((df['log_casos_slope_4'] > 0) & (df['dist_to_typical_peak'] > 8)).astype(int)
        if 'log_casos_slope_8' in df.columns:
            df['slope8_x_dist_peak'] = df['log_casos_slope_8'] * df['dist_to_typical_peak']

        # gating lag52 x slope: lag sazonal só deve puxar para cima se slope atual confirma yoy positivo
        if 'log_casos_lag52' in df.columns and 'log_casos_slope_4' in df.columns:
            df['lag52_x_slope4'] = df['log_casos_lag52'] * df['log_casos_slope_4']
        if 'log_casos_yoy52' in df.columns and 'log_casos_slope_4' in df.columns:
            df['yoy52_x_slope4'] = df['log_casos_yoy52'] * df['log_casos_slope_4']
            # confirmação dupla: yoy positivo + subindo => forte sinal tardio
            df['yoy_positive_rising'] = ((df['log_casos_yoy52'] > 0) & (df['log_casos_slope_4'] > 0)).astype(int)

        # momentum ajustado para pico esperado: lag52 + momentum * weeks_to_peak (projeção linear)
        if 'log_casos_lag52' in df.columns and 'log_casos_momentum_4' in df.columns:
            # momentum semanal médio
            mom_per_week = df['log_casos_momentum_4'] / 4.0
            df['trended_peak_lag52'] = df['log_casos_lag52'] + mom_per_week * df['weeks_to_typical_peak'].clip(lower=0)

        # flag pico tardio histórico: se pico anterior foi tardio (>S12), aumentar prior tardio
        # usa weeks_since_peak_52: se >20 e ainda subindo => tardio
        if 'weeks_since_peak_52' in df.columns and 'log_casos_slope_4' in df.columns:
            df['late_peak_risk'] = ((df['weeks_since_peak_52'] > 20) & (df['log_casos_slope_4'] > 0)).astype(int)
            df['weeks_since_peak_x_slope'] = df['weeks_since_peak_52'] * df['log_casos_slope_4'].fillna(0)

        logger.info("expand_peak_timing: distância ao pico típico + gating slope adicionados")
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
        logger.info("Starting advanced feature engineering (EXP v2)...")
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

        # ============================================================
        # EXPANSAO 4 (expand_climatology - v2): media 3y e anomalias vs clim
        # ============================================================
        if self.expand_climatology:
            df = self.create_climatology_features(df)
            # Interação climatológica × sazonalidade (pós-climatologia): anomalia vs estação
            try:
                if 'temp_mean_mean_anomaly_clim' in df.columns and 'sin_semana' in df.columns:
                    # usa lag1 se existir (evita leak), senão valor atual já é shift-safe via climatologia lag52
                    col = 'temp_mean_mean_anomaly_clim_lag1' if 'temp_mean_mean_anomaly_clim_lag1' in df.columns else 'temp_mean_mean_anomaly_clim'
                    df['temp_anomaly_clim_x_sin'] = df[col] * df['sin_semana']
                    df['temp_anomaly_clim_x_cos'] = df[col] * df['cos_semana']
            except Exception:
                pass

        # ============================================================
        # EXPANSAO 5 (expand_trend - v2): slopes, momentum, CV, accel
        # ============================================================
        if self.expand_trend:
            df = self.create_trend_features(df)

        # ============================================================
        # EXPANSAO 6 (expand_outbreak - v2): fase epidemica e contagens
        # ============================================================
        if self.expand_outbreak:
            df = self.create_outbreak_features(df)
        # ============================================================
        # EXPANSAO 7 (expand_regime_conditional - v3): lags condicionais + gating anti-phantom
        # ============================================================
        if self.expand_regime_conditional:
            df = self.create_regime_conditional_features(df)
        # ============================================================
        # EXPANSAO 8 (expand_peak_timing - v3): distancia ao pico tipico + gating slope
        # ============================================================
        if self.expand_peak_timing:
            df = self.create_peak_timing_features(df)

        if legacy_mode or convention == 'legacy':
            logger.warning("legacy_mode nao suportado no modo EXP (use convention='advanced')")

        if se_col is not None:
            df['SE'] = se_col

        df, feature_cols = self.select_features(df, 'target', all_horizons=target_horizons)

        if se_col is not None and 'SE' not in df.columns:
            df['SE'] = se_col

        logger.info(f"Feature engineering complete (EXP v2). Shape: {df.shape}, Features: {len(feature_cols)}")
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
