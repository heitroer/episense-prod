"""Teste retrospectivo de adaptação local e transferência direta de um LightGBM reduzido."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.observations import observed_values
from data.epiweeks import add_epiweeks, date_to_epiweek
from models.metrics import QUANTILES, paired_forecast_metrics
from models.validation_calendar import label_available_week

CONTROL = "5002704_dengue"
SERIES = (CONTROL, "5003702_dengue", "5002704_chikungunya")
FOLDS = ("202236", "202336", "202436", "202537")
HORIZONS = (1, 2, 4, 8)


def series_in(directory):
    if (directory / "5003207_dengue.csv").exists():
        return (*SERIES, "5003207_dengue")
    return SERIES


def features(path, case_delay):
    frame = pd.read_csv(path, dtype={"SE": str}).sort_values("SE").set_index("SE")
    dates = pd.to_datetime(frame.data_iniSE)
    if (frame.index.has_duplicates or not dates.diff().dropna().eq(pd.Timedelta(weeks=1)).all()
            or list(dates.map(date_to_epiweek)) != list(frame.index)):
        raise ValueError(f"Calendário inválido em {path}")
    if not np.isfinite(frame[["pop", "casos"]]).all().all() or min(frame["pop"]) <= 0 or min(frame.casos) < 0:
        raise ValueError(f"Contagem ou população inválida em {path}")
    rate = (100000 * frame.casos / frame["pop"]).shift(case_delay)
    x = pd.DataFrame(index=frame.index)
    for lag in (0, 1, 2, 4, 8, 52):
        x[f"rate_lag{lag}"] = np.log1p(rate.shift(lag))
    for window in (4, 8):
        x[f"rate_mean{window}"] = np.log1p(rate.rolling(window).mean())
    angle = 2*np.pi*frame.index.str[-2:].astype(int)/52.18
    x["sin"], x["cos"] = np.sin(angle), np.cos(angle)
    for column in ("tempmed", "umidmed"):
        x[column] = frame[column]
    return frame, x


def run(directory, cutoff, case_delay):
    data = {k: features(directory / (k + ".csv"), case_delay) for k in series_in(directory)}
    calendar = data[CONTROL][0].index
    if any(not frame.index.equals(calendar) for frame, _ in data.values()):
        raise ValueError("Séries com calendários distintos")
    rows = []
    for fold_idx, fold in enumerate(FOLDS):
        end = FOLDS[fold_idx+1] if fold_idx+1 < len(FOLDS) else add_epiweeks(fold, 52)
        for h in HORIZONS:
            ready = np.array([label_available_week(add_epiweeks(se, h), 16) for se in calendar])
            train = (ready <= fold) & (np.arange(len(calendar)) >= 52 + case_delay)
            test = (calendar >= fold) & (calendar < end) & (ready <= cutoff)
            if train.sum() < 100 or not test.any():
                raise ValueError("Janela insuficiente")
            trained = {}
            for k, (frame, x) in data.items():
                # Denominador é a população conhecida na origem, inclusive na conversão inversa.
                y = np.log1p(100000 * observed_values(frame).shift(-h) / frame["pop"])
                fit = train & y.notna()
                if fit.sum() < 100:
                    raise ValueError(f"Desfechos observados insuficientes em {k}, H{h}")
                trained[k] = []
                for tau in QUANTILES:
                    params = dict(objective="quantile", alpha=tau, num_leaves=15, learning_rate=.04,
                                  seed=42, verbosity=-1, num_threads=1, force_col_wise=True)
                    model = lgb.train(params, lgb.Dataset(x.loc[fit], label=y.loc[fit]),
                                      num_boost_round=150)
                    trained[k].append(model)
            for k, (frame, x) in data.items():
                observed = observed_values(frame)
                eligible = test & observed.shift(-h).notna() & observed.shift(case_delay).notna()
                if not eligible.any():
                    raise ValueError(f"Nenhum par observado em {k}, H{h}")
                for method, source in (("local", k), ("transfer", CONTROL)):
                    if method == "transfer" and k == CONTROL:
                        continue
                    prediction = np.stack([m.predict(x.loc[eligible], num_threads=1) for m in trained[source]], axis=1)
                    prediction = np.maximum(0, np.expm1(np.sort(prediction, axis=1)))
                    prediction = np.rint(prediction * frame.loc[eligible, "pop"].to_numpy()[:, None]/100000)
                    for origin, values in zip(calendar[eligible], prediction):
                        target = add_epiweeks(origin, h)
                        rows.append(dict(series=k, method=method, horizon=h, fold=fold, origin=origin,
                                         target=target, actual=frame.at[target,"casos"],
                                         persistence=frame.at[add_epiweeks(origin, -case_delay),"casos"],
                                         **{str(t):v for t,v in zip(QUANTILES, values)}))
    predictions = pd.DataFrame(rows)
    def metrics(group):
        result = paired_forecast_metrics(group.actual,group[[str(t) for t in QUANTILES]],group.persistence)
        if result["n_scored"] != len(group) or group.origin.duplicated().any():
            raise ValueError("Previsões inválidas ou duplicadas")
        return {k:result[k] for k in ("n_scored","mae","persistence_mae","wis_total","coverage_90")}
    summary = []
    for (series, h, method), group in predictions.groupby(["series","horizon","method"]):
        control_origins = predictions.loc[(predictions.series==CONTROL) & (predictions.horizon==h),"origin"]
        if set(group.origin) != set(control_origins):
            raise ValueError("Origens divergentes")
        summary.append(dict(series=series, horizon=int(h), method=method, **metrics(group)))
    predictions.to_csv(directory/f"expansion_predictions_lag{case_delay}.csv", index=False)
    hashes = {k:hashlib.sha256((directory/(k+".csv")).read_bytes()).hexdigest() for k in series_in(directory)}
    (directory/f"expansion_results_lag{case_delay}.json").write_text(json.dumps(
        dict(cutoff=cutoff, case_delay=case_delay, sha256=hashes, metrics=summary), indent=2, allow_nan=False))
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cutoff", default="202646", help="SE da avaliação")
    parser.add_argument("--case-delay", type=int, choices=(0, 17), default=17,
                        help="Atraso conservador de 17 semanas ou sensibilidade consolidada")
    args = parser.parse_args()
    run(args.directory, args.cutoff, args.case_delay)
