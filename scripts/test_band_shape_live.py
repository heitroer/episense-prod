"""
Teste de integração EXP-Q2: fan ANTES vs DEPOIS da band_shape (caminho live).
Usa episense_base.csv + EpisenseInference.predict exatamente como API/dashboard.
"""
import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.inference import EpisenseInference  # noqa: E402


def fan(inference, df):
    preds = inference.predict(df)
    out = {}
    for h in sorted(preds.keys()):
        row = {"h": h}
        for tau, arr in preds[h].items():
            if getattr(arr, 'size', 0) == 0:
                continue
            val = float(arr[-1])
            if np.isnan(val):
                continue
            cases = float(np.expm1(val))
            row[f"q{int(tau*100):02d}"] = int(round(cases))
        out[h] = row
    return out


def main():
    with open(ROOT / 'config/config.yaml') as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(ROOT / 'data/processed/episense_base.csv')
    df['SE'] = df['SE'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    print(f"base: {len(df)} rows, última SE {df['SE'].max()}")

    # configuração ATUAL (band_shape enabled conforme config.yaml)
    inf_on = EpisenseInference(str(ROOT / 'models'), copy.deepcopy(cfg))

    # config com band_shape OFF (estado antigo) + viés apenas na mediana (estado antigo 1.08)
    cfg_old = copy.deepcopy(cfg)
    cfg_old['postprocessing']['band_shape']['enabled'] = False
    inf_off = EpisenseInference(str(ROOT / 'models'), cfg_old)

    fan_old = fan(inf_off, df.copy())
    fan_new = fan(inf_on, df.copy())

    print(f"\n{'h':>3} | {'ANTES (raw)':^45} | {'DEPOIS (band_shape)':^45} | ratios f95 antes->depois")
    print(f"{'':>3} | {'q05':>6}{'q25':>7}{'q50':>7}{'q75':>7}{'q95':>7} | "
          f"{'q05':>6}{'q25':>7}{'q50':>7}{'q75':>7}{'q95':>7}")
    for h in sorted(fan_old.keys()):
        o, n = fan_old[h], fan_new.get(h, {})
        def g(d, k):
            return d.get(k, 0)
        med_o = max(g(o, 'q50'), 1e-9)
        med_n = max(g(n, 'q50'), 1e-9)
        f95_o = g(o, 'q95') / med_o
        f95_n = g(n, 'q95') / med_n
        f05_o = g(o, 'q05') / med_o
        f05_n = g(n, 'q05') / med_n
        print(f"{h:>3} | {g(o,'q05'):>6}{g(o,'q25'):>7}{g(o,'q50'):>7}{g(o,'q75'):>7}{g(o,'q95'):>7} | "
              f"{g(n,'q05'):>6}{g(n,'q25'):>7}{g(n,'q50'):>7}{g(n,'q75'):>7}{g(n,'q95'):>7} | "
              f"f95 {f95_o:.2f}->{f95_n:.2f}  f05 {f05_o:.2f}->{f05_n:.2f}")

    # shape-corr antes/depois
    def corr(seq_lo, seq_med):
        import numpy as _np
        a = _np.log1p(_np.array(seq_lo, dtype=float))
        b = _np.log1p(_np.array(seq_med, dtype=float))
        return float(_np.corrcoef(a, b)[0, 1])
    hs = [h for h in fan_old]
    for d, name, lo in [(fan_old, 'ANTES', 'q05'), (fan_new, 'DEPOIS', 'q05')]:
        c05 = corr([d[h][lo] for h in hs], [d[h]['q50'] for h in hs])
        c95 = corr([d[h]['q95'] for h in hs], [d[h]['q50'] for h in hs])
        print(f"{name}: shape_corr q05xq50={c05:.3f} q95xq50={c95:.3f}")


if __name__ == '__main__':
    main()
