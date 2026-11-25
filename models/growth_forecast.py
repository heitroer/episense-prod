"""Experimental municipal and regional models for retrospective dengue forecasting."""

import numpy as np
import pandas as pd

from data.epiweeks import add_epiweeks
from data.observations import observed_values
from models.metrics import QUANTILES

LEVELS = np.asarray(QUANTILES)


class GrowthModel:
    """Seasonal robust ridge regression of future log growth, with empirical forecast errors."""

    def __init__(self, frame, peer, horizon, maturity):
        self.h = horizon
        self.delay = horizon + maturity + 1
        counts = observed_values(frame).astype(float)
        # Smooth weekly reporting noise before calculating growth.
        self.level = np.log(counts.rolling(3, min_periods=1).mean() + 2).to_numpy()
        self.target = np.log(counts.shift(-horizon).to_numpy() + 2)
        logs = np.log(counts + 2)
        recent = np.log((counts + counts.shift(1) + 4) / 2)
        prior = np.log(counts.shift(2).rolling(3).mean() + 2)
        slow = np.log(counts.shift(5).rolling(5).mean() + 2)
        angle = 2 * np.pi * (frame.SE.astype(int) % 100) / 52.18
        # A one-week lag permits asynchronous municipal reporting.
        other = peer.reindex(frame.SE).reset_index(drop=True).shift(1)
        regional = np.log(other.rolling(4).mean() + 2)
        regional_growth = (regional - regional.shift(5)).fillna(0)
        anomaly = regional - regional.shift(10).rolling(52, min_periods=12).median()
        self.features = np.column_stack(
            [
                self.level,
                recent - prior,
                prior - slow,
                logs - recent,
                np.sin(angle),
                np.cos(angle),
                regional_growth,
                anomaly.fillna(0),
            ]
        )
        self.valid = np.isfinite(self.features).all(axis=1)
        season = np.column_stack([np.sin(angle), np.cos(angle)])
        # Use overlapping and disjoint windows to estimate the slope even when a
        # holiday displaces notifications by one or two weeks.
        local = [
            np.log(counts.shift(lag).rolling(width).mean() + 2)
            for lag, width in [
                (0, 2),
                (2, 3),
                (5, 4),
                (9, 5),
                (14, 7),
                (21, 10),
                (31, 16),
            ]
        ]
        series = np.column_stack(local)
        slopes = np.clip(series[:, :-1] - series[:, 1:], -1.6, 1.6)
        other_levels = [
            np.log(other.shift(lag).rolling(width).mean() + 2)
            for lag, width in [(0, 3), (3, 5), (8, 8), (16, 12)]
        ]
        other_values = np.column_stack(other_levels)
        peer_slopes = np.nan_to_num(
            np.clip(other_values[:, :-1] - other_values[:, 1:], -1.6, 1.6)
        )
        intensity = series[:, 0] - np.log(counts.shift(20).rolling(52).mean() + 2)
        state = np.column_stack([self.level - 4, slopes, peer_slopes, intensity])
        self.design = np.column_stack(
            [
                np.ones(len(frame)),
                state,
                logs - recent,
                season,
                np.sin(2 * angle),
                np.cos(2 * angle),
                (state[:, :, None] * season[:, None, :]).reshape(len(frame), -1),
            ]
        )
        self.valid &= np.isfinite(self.design).all(axis=1)
        self.errors = np.full(len(counts), np.nan)
        self.coefficients = {}
        # Generate calibration forecasts in chronological 13-week blocks.
        for first in range(182, len(counts), 13):
            idx = np.arange(first, min(first + 13, len(counts)))
            idx = idx[self.valid[idx]]
            pred = self.center(idx, first)
            self.errors[idx] = np.exp(self.target[idx] * 0.25) - np.exp(pred * 0.25)

    def center(self, idx, cutoff):
        if cutoff not in self.coefficients:
            stop = cutoff - self.delay + 1
            keep = self.valid[:stop] & np.isfinite(self.target[:stop])
            features = self.design[:stop][keep]
            response = (self.target - self.level)[:stop][keep]
            w = 0.5 ** ((cutoff - np.flatnonzero(keep)) / 260)
            penalty = np.eye(features.shape[1]) * 9
            penalty[0, 0] = 0.01
            # Robust reweighting prevents exceptional historical seasons from dominating.
            robust = w
            for _ in range(3):
                beta = np.linalg.solve(
                    features.T @ (robust[:, None] * features) + penalty,
                    features.T @ (robust * response),
                )
                resid = np.abs(response - features @ beta)
                robust = w * np.minimum(
                    1, (0.25 + 0.06 * self.h) / np.maximum(0.01, resid)
                )
            self.coefficients[cutoff] = beta
        return self.level[idx] + self.design[idx] @ self.coefficients[cutoff]

    def predict(self, i, cutoff):
        center = self.center(i, cutoff)
        stop = cutoff - self.delay + 1
        keep = np.isfinite(self.errors[:stop])
        errors = self.errors[:stop][keep]
        delta = self.features[:stop][keep] - self.features[i]
        distance = np.sum((delta / [1.8, 1, 1.2, 2, 1, 1, 1.5, 2.5]) ** 2, axis=1)
        w = np.exp(-0.5 * np.minimum(distance, 60))
        w *= 0.5 ** ((cutoff - np.flatnonzero(keep)) / 156)
        order = errors.argsort()
        w = w[order]
        offsets = np.interp(LEVELS, (w.cumsum() - 0.5 * w) / w.sum(), errors[order])
        band = np.maximum(0, np.exp(center * 0.25) + offsets) ** 4 - 2
        return np.maximum(0, np.rint(band))

    def regional_agreement(self, i):
        local_slope = np.median(self.design[i, 2:6])
        peer_slope = self.features[i, 6]
        return np.clip(local_slope / 0.45, 0, 1) * np.clip(peer_slope / 0.35, 0, 1)


class PanelGrowth:
    """Quantis de crescimento treinados com trajetórias municipais contemporâneas."""

    def __init__(self, frame, panel, city, h, maturity):
        import lightgbm as lgb

        self.models = {}
        self.cache = {}
        self.lgb = lgb
        self.h = h
        self.maturity = maturity
        self.city = city
        self.weeks = frame.SE.to_numpy()
        self.data = []
        grid = panel.index
        season = 2 * np.pi * (grid.astype(int) % 100) / 52.18
        for name, counts in panel.items():
            # Reference excludes the city being predicted, even during peer training.
            neighbors = panel.drop(columns=name)
            regional = neighbors.div(
                neighbors.shift(6).rolling(104, min_periods=40).mean() + 2
            )
            pressure = np.log(regional.median(axis=1).shift(1).rolling(4).mean() + 0.1)
            level = np.log(counts.rolling(3).mean() + 2)
            slow = np.log(counts.shift(3).rolling(6).mean() + 2)
            background = np.log(counts.shift(8).rolling(26).mean() + 2)
            x = pd.DataFrame(
                {
                    "season_sin": np.sin(season),
                    "season_cos": np.cos(season),
                    "scale": level,
                    "season_intensity": level - background,
                    "momentum": level - slow,
                    "recent": np.log(counts + 2) - level,
                    "momentum2": slow - np.log(counts.shift(9).rolling(8).mean() + 2),
                    "pressure": pressure,
                    "regional_growth": pressure - pressure.shift(4),
                    "short_growth": level - level.shift(2),
                    "last_season": level - level.shift(52),
                    "recent_season": background
                    - np.log(counts.shift(34).rolling(26).mean() + 2),
                },
                index=grid,
            )
            # The additive shift is part of the prediction, including at test time.
            anchor = 0.85 * level + 0.15 * background
            y = np.log(counts.shift(-h) + 2) - anchor
            self.data.append((name, x, y, anchor))
            if name == city:
                self.x, self.anchor = x, anchor

    def predict(self, i, cutoff):
        week = self.weeks[cutoff]
        if week not in self.models:
            last_week = add_epiweeks(week, -self.h - self.maturity - 1)
            xs, ys, ws = [], [], []
            for name, x, y, _ in self.data:
                keep = (x.index <= last_week) & y.notna() & x.scale.notna()
                xs.append(x[keep])
                ys.append(y[keep])
                weight = 5 if name == self.city else 1
                ws.append(np.full(keep.sum(), weight))
            xt, yt, wt = pd.concat(xs), pd.concat(ys), np.concatenate(ws)
            self.models[week] = []
            for q in QUANTILES:
                params = {
                    "objective": "quantile",
                    "alpha": q,
                    "learning_rate": 0.04,
                    "num_leaves": 12,
                    "min_data_in_leaf": 40,
                    "lambda_l2": 3,
                    "num_threads": 1,
                    "verbosity": -1,
                    "seed": 42,
                }
                model = self.lgb.train(
                    params,
                    self.lgb.Dataset(xt, label=yt, weight=wt),
                    num_boost_round=120,
                )
                self.models[week].append(model)
        # Predict all weeks in one matrix call per quantile and cutoff.
        if week not in self.cache:
            x = self.x.loc[self.weeks]
            raw = np.column_stack(
                [m.predict(x, num_threads=1) for m in self.models[week]]
            )
            self.cache[week] = np.maximum(
                0,
                np.rint(
                    np.exp(
                        self.anchor.loc[self.weeks].to_numpy()[:, None]
                        + np.sort(raw, axis=1)
                    )
                    - 2
                ),
            )
        return self.cache[week][i]


def combine_bands(municipal, regional, growth, agreement):
    """Combine extrapolation with two tree models, in the same order at every origin."""
    w = 0.30 + 0.40 * agreement
    other = (1 - w) / 2
    bands = np.asarray([municipal, regional, growth], dtype=float)
    log_bands = np.log(bands + 2)
    center = other * (log_bands[0, 2] + log_bands[1, 2]) + w * log_bands[2, 2]
    # Combine widths independently. A surge can shift the center without
    # forcing the largest upper tail to dominate the whole distribution.
    log_pool = np.mean(log_bands, axis=0)
    spread = np.exp(log_pool / 2) - np.exp(log_pool[2] / 2)
    return np.maximum(0, np.maximum(0, np.exp(center / 2) + spread) ** 2 - 2)
