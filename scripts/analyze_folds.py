"""
EXP-Q2 — previsões históricas do walk-forward POR FOLD: retreino novo (models_q2)
vs produção canônica (models), a partir de models*/validation_predictions.json
(blocos por h×tau×fold). Métricas: MAE (q50), cov90, cov50, WIS (models.metrics).
Sanidade de forma da banda (raw): f95 = q95/med, taxa de q05 "no chão".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.metrics import weighted_interval_score, coverage_interval  # noqa: E402

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load_vp(vp_dir: Path):
    with open(vp_dir / 'validation_predictions.json') as f:
        return json.load(f)


def collect(vp):
    """{(fold, se, h): {'y': float, (h,tau): pred}} de todos os horizontes."""
    folds = {}
    for h in range(1, 9):
        d = vp[str(h)]
        for tau in QUANTILES:
            key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
            for blk in d[key]:
                fold = int(blk['fold'])
                for se, yt, yp in zip(blk['test_se'], blk['y_test_casos'], blk['y_pred_casos']):
                    e = folds.setdefault((fold, str(se).zfill(6), int(h)), {})
                    if tau == 0.50:
                        e['y'] = float(yt)
                    e[tau] = float(yp)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='logs/experimentos-arquivados/modelos/models_q2')
    ap.add_argument('--baseline', default='models')
    args = ap.parse_args()

    cand = collect(load_vp(ROOT / args.candidate))
    prod = collect(load_vp(ROOT / args.baseline))

    def metrics(fold_rows, h):
        ys, q = [], {t: [] for t in QUANTILES}
        for k, r in sorted(fold_rows.items()):
            if r.get('h') != h:  # noqa
                continue
            ys.append(r['y'])
            for t in QUANTILES:
                q[t].append(max(r.get((h, t), np.nan), 0.0))
        ys = np.array(ys); q = {t: np.array(v) for t, v in q.items()}
        if len(ys) == 0 or np.all(np.isnan(q[0.50])):
            return None
        return {
            'n': len(ys),
            'se_min': min(fold_rows), 'se_max': max(fold_rows),
            'mae': float(np.mean(np.abs(ys - q[0.50]))),
            'cov90': coverage_interval(ys, q[0.05], q[0.95]),
            'cov50': coverage_interval(ys, q[0.25], q[0.75]),
            'wis': weighted_interval_score(ys, q),
            'f95_med': float(np.quantile(q[0.95] / np.maximum(q[0.50], 0.5), 0.5)),
            'q05_zero': float(np.mean(q[0.05] < 0.5)),
        }

    FOLDS = sorted(set(f for f, _, _ in cand) & set(f for f, _, _ in prod))

    def row_metrics(rows_all, fold, h):
        ys, q = [], {t: [] for t in QUANTILES}
        sems = set()
        for (f_, se, h_), r in rows_all.items():
            if f_ not in FOLDS or (fold is not None and f_ != fold) or (h is not None and h_ != h):
                continue
            sems.add(se)
            ys.append(r['y'])
            for t in QUANTILES:
                q[t].append(max(r.get(t, np.nan), 0.0))
        ys = np.array(ys); q = {t: np.array(v) for t, v in q.items()}
        if len(ys) == 0 or np.all(np.isnan(q[0.50])):
            return None
        return {
            'n': len(ys), 'se_min': min(sems), 'se_max': max(sems),
            'mae': float(np.mean(np.abs(ys - q[0.50]))),
            'cov90': coverage_interval(ys, q[0.05], q[0.95]),
            'cov50': coverage_interval(ys, q[0.25], q[0.75]),
            'wis': weighted_interval_score(ys, q),
            'f95_med': float(np.quantile(q[0.95] / np.maximum(q[0.50], 0.5), 0.5)),
            'q05_zero': float(np.mean(q[0.05] < 0.5)),
        }

    print(f"{'fold':>4} | {'semanas (SE)':>16} | {'n':>3} | "
          f"{'MAE cand':>8} {'MAE prod':>8} | {'cov90 cand':>10} {'cov90 prod':>10} | "
          f"{'cov50 cand':>10} {'cov50 prod':>10} | {'WIS cand':>8} {'WIS prod':>8}")
    for f in [*FOLDS, None]:
        c = row_metrics(cand, f, None)
        p = row_metrics(prod, f, None)
        if c is None or p is None:
            continue
        values = " | ".join(f"{c[k]:8.3f} {p[k]:8.3f}" for k in ("mae", "cov90", "cov50", "wis"))
        label = "AGG" if f is None else str(f)
        print(f"{label:>4} | {c['se_min']}–{c['se_max']} | {c['n']:3} | {values}")

    # por horizonte (mae) — cand vs prod
    print("\nMAE por horizonte (cand → prod):")
    for h in range(1, 9):
        c, p = row_metrics(cand, None, h), row_metrics(prod, None, h)
        if not c or not p:
            continue
        mc, mp = c["mae"], p["mae"]
        print(f"  h{h}: {mc:8.2f} → {mp:8.2f}   ({(mc / mp - 1) * +100:+6.1f}%)")

    # sanidade de forma da banda crua (cand): f95 e chão do q05
    print("\nForma da banda crua do candidato (f95 mediano por h; q05<0.5 casos):")
    for h in range(1, 9):
        summary = row_metrics(cand, None, h)
        if not summary:
            continue
        f95, z = summary["f95_med"], summary["q05_zero"]
        print(f"  h{h}: f95 {f95:5.2f}x | q05 zero {z:5.1%}")


if __name__ == '__main__':
    main()
