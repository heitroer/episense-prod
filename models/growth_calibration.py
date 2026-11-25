"""Chronological calibration of the municipal dengue forecast ensemble."""

import numpy as np
import pandas as pd

from data.observations import observed_values

QCOLS = ["q05", "q25", "q50", "q75", "q95"]


def weighted_q(values, weights, quantiles):
    order = np.argsort(values)
    w = weights[order]
    return np.interp(quantiles, (w.cumsum() - 0.5 * w) / w.sum(), values[order])


def replay_calibration(components, frame, maturity):
    """Replay issuance in chronological order with outcomes gated by availability."""
    rows = components
    medians_by_method = rows.pivot(
        index=["origin", "h"], columns="method", values="q50"
    ).sort_index()
    expert_medians = np.log(
        medians_by_method[["municipal", "regional", "tendencia"]].to_numpy() + 4
    )
    rows = rows[rows.method == "combinado"].copy().sort_values(["origin", "h"])
    index = pd.Series(np.arange(len(frame)), index=frame.SE.astype(int))
    origin = index.loc[rows.origin.astype(int)].to_numpy()
    ready = rows.available.astype(int).to_numpy()
    dates = rows.origin.astype(int).to_numpy()
    h = rows.h.to_numpy()
    y = rows.y.to_numpy()
    angle = 2 * np.pi * (dates % 100) / 52.18
    c = observed_values(frame)
    log_counts = np.log(c + 4)
    level = np.log(c.rolling(4).mean() + 4)
    noise = log_counts.diff().abs().rolling(13).median().to_numpy()[origin]
    change = (level - level.shift(5)).to_numpy()[origin]
    volatility = np.column_stack(
        [(level - level.shift(k)).abs().rolling(26).median() for k in range(1, 9)]
    )[origin, h - 1]
    state = np.column_stack(
        [
            np.log(h),
            level.to_numpy()[origin],
            change,
            noise,
            np.sin(angle),
            np.cos(angle),
        ]
    )
    raw = rows[QCOLS].to_numpy()
    log_band = np.log(raw + 4)
    center = log_band[:, 2].copy()
    ylog = np.log(y + 4)
    fast = np.log(c.rolling(2).mean() + 4)
    prior = np.log(c.shift(2).rolling(4).mean() + 4)
    trend = np.clip(((fast - prior) / 3).to_numpy()[origin], -0.22, 0.22)
    damped = fast.to_numpy()[origin] + 3 * (1 - np.exp(-h / 3)) * trend
    expert_medians = np.column_stack([expert_medians, damped])
    loss = np.abs(expert_medians - ylog[:, None])
    skew = (
        np.clip(
            (log_band[:, 4] - center) / np.maximum(0.01, center - log_band[:, 0]),
            0.5,
            2,
        )
        ** 0.4
    )
    power = 0.50
    transformed = (raw + 4) ** power
    raw_spread = (transformed[:, 4] - transformed[:, 0]) / 3.29
    signal = power * (raw[:, 2] + 4) ** power * (0.08 + noise + 0.65 * volatility)
    scale = np.sqrt(np.maximum(0.15, raw_spread) * signal)
    scores = np.full(len(rows), np.nan)
    signed = np.full(len(rows), np.nan)
    results = np.zeros(raw.shape)
    raw_meds = center.copy()
    integral = np.zeros((8, 2))
    previous_date = np.full(8, -1)
    count = np.zeros(len(rows), dtype=int)
    latest = np.zeros(len(rows), dtype=int)
    for j in range(len(rows)):
        mature = ready <= dates[j]
        count[j] = mature.sum()
        if count[j]:
            latest[j] = ready[mature].max()
        if count[j] < 48:
            results[j] = raw[j]
        else:
            delta = (state[mature] - state[j]) / [0.43, 0.95, 0.7, 0.27, 0.9, 0.9]
            age = origin[j] - origin[mature] - h[mature] - 1 - maturity
            weights = np.exp(-0.5 * np.minimum(60, (delta**2).sum(axis=1))) * 0.5 ** (
                age / 60
            )
            risk = weights @ loss[mature] / weights.sum()
            mix = np.exp(-4 * (risk - risk.min()) / max(0.15, risk.mean()))
            mix /= mix.sum()
            conditional = weighted_q(ylog[mature] - raw_meds[mature], weights, 0.5)
            center[j] = (
                0.60 * raw_meds[j]
                + 0.40 * (mix @ expert_medians[j])
                + 0.15 * conditional
            )
            # Coverage feedback from resolved forecasts. Pool nearby horizons to
            # avoid pretending eight forecasts of the same target are independent.
            near = np.exp(
                -0.5 * ((np.log(h[mature]) - np.log(h[j])) / 0.45) ** 2
            ) * 0.5 ** (age / 32)
            covered = np.column_stack(
                [
                    (y[mature] >= results[mature, 1])
                    & (y[mature] <= results[mature, 3]),
                    (y[mature] >= results[mature, 0])
                    & (y[mature] <= results[mature, 4]),
                ]
            )
            target = np.array([0.5, 0.9])
            recent_coverage = (near @ covered + 20 * target) / (near.sum() + 20)
            h_index = h[j] - 1
            newly = ready[mature] > previous_date[h_index]
            for week in np.unique(ready[mature][newly]):
                recent = ready[mature] == week
                hw = np.exp(-0.5 * ((h[mature][recent] - h[j]) / 1.25) ** 2)
                coverage = np.average(covered[recent], axis=0, weights=hw)
                integral[h_index] = np.clip(
                    0.992 * integral[h_index] + 0.028 * (target - coverage),
                    [-0.12, -0.12],
                    [0.12, 0.07],
                )
            previous_date[h_index] = dates[j]
            probabilities = np.clip(
                target + 0.8 * (target - recent_coverage) + integral[h_index],
                [0.32, 0.74],
                [0.68, 0.98],
            )
            # Several horizons can share an observed target. Use distinct
            # target weeks to estimate the effective sample for tail ranks.
            mass = np.bincount(origin[mature] + h[mature], weights=weights)
            effective_weeks = mass.sum() ** 2 / (mass @ mass)
            probabilities = np.minimum(
                0.985, probabilities * (1 + 1 / max(8, effective_weeks))
            )
            inner, outer = weighted_q(scores[mature], weights, probabilities)
            offsets = np.array(
                [
                    -outer / skew[j],
                    -inner / skew[j],
                    0,
                    inner * skew[j],
                    outer * skew[j],
                ]
            )
            tails = weighted_q(
                signed[mature],
                weights,
                [
                    (1 - probabilities[1]) / 2,
                    (1 - probabilities[0]) / 2,
                    0.5,
                    (1 + probabilities[0]) / 2,
                    (1 + probabilities[1]) / 2,
                ],
            )
            tails[2] = 0
            tails[:2] = np.minimum(0, tails[:2])
            tails[3:] = np.maximum(0, tails[3:])
            asym_share = effective_weeks / (effective_weeks + 30)
            offsets = (1 - asym_share) * offsets + asym_share * tails
            results[j] = np.maximum(
                0,
                np.rint(
                    np.maximum(0, np.exp(center[j] * power) + scale[j] * offsets)
                    ** (1 / power)
                    - 4
                ),
            )
        # Stored scores become accessible only through the maturity mask above.
        residual = (y[j] + 4) ** power - np.exp(center[j] * power)
        signed[j] = residual / scale[j]
        scores[j] = (
            abs(residual) / scale[j] * (skew[j] if residual < 0 else 1 / skew[j])
        )
    rows[QCOLS] = results
    rows["calibration_n"] = count
    rows["last_calibration_available_se"] = latest
    return rows[rows.phase == "evaluation"]
