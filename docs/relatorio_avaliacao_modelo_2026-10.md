# Relatório de Avaliação do Modelo — EpiSense CG-MS (out/2026)

Ensemble LightGBM quantil (5 seeds), horizontes h1–h8, validação walk-forward
epi 2022–2025 (gap 8). Métricas: `mae`, `rmse`, `bias`, `wis_total`,
`wis_50`, `wis_90`, `width_50/90`, `coverage_50/90` (+ baseline sazonal).

Ferramenta: análise ad-hoc sobre `models/validation_results.json` e
`models/validation_predictions.json` (séries desde fold0 = SE 202236).

## Período de cada fold (contexto epidemiológico)

| Fold | Temporada | Pico real | Volume total | Caracter |
|---|---|---|---|---|
| 0 | 202236–202335 | **1.132** (SE 202314) | 17.645 | Surto grande |
| 1 | 202336–202435 | **769** (SE 202407) | 12.261 | Surto moderado |
| 2 | 202436–202536 | 222 | 5.960 | Ano fraco |
| 3 | 202537–202633 | **119** (SE 202612) | 2.329 | Ano muito fraco |

4 folds com regimes de amplitude 10× entre si — isso define quase todos os problemas.

## Métricas por fold × horizonte

**MAE (mediana):**

| fold | h1 | h2 | h4 | h8 |
|---|---|---|---|---|
| 0 | 64,7 | 104,8 | 153,1 | 187,4 |
| 1 | 48,5 | 57,7 | 75,9 | 88,9 |
| 2 | 17,6 | 18,4 | 24,1 | 18,2 |
| 3 | 15,1 | 30,7 | 41,6 | 63,9 |

**Bias (mediana − real; + = superestima):**

| fold | h1 | h2 | h4 | h8 |
|---|---|---|---|---|
| 0 | −43,5 | −86,2 | −136,2 | −168 |
| 1 | −25,7 | −19,1 | −22,2 | −34,6 |
| 2 | −4,0 | −5,6 | −3,0 | −2,7 |
| 3 | +9,3 | +26,6 | +38,1 | +57,8 |

**Cov90 / largura IC90:**

| fold | h1 | h4 | h8 |
|---|---|---|---|
| 0 | 0,81 / w330 | 0,87 / w602 | 0,75 / w792 |
| 1 | 0,77 / w250 | 0,88 / w470 | 0,88 / w435 |
| 2 | 0,89 / w108 | 1,00 / w362 | 0,96 / w473 |
| 3 | 0,80 / w87 | 0,66 / w194 | 0,53 / w296 |

## Pontos fracos (por gravidade)

### P1 — Subestimação dos picos (não extrapolação de árvores)
Mediana no pico de 2023: h1 830 vs 1.132 (−27%); h4 **461 vs 1.132 (−59%)**;
h8 535 vs 1.132 (−53%). Bins ≥400 casos: bias −95 (h1) → −296 (h8).

### P2 — Cobertura condicional errada (infla calma, aperta crise)
- fold2: cov90 = 1,00 (h4) com banda 362 casos em ano de pico 222 (banda 2–4× o nível);
- fold3: cov90 h8 = 0,53, **47% das semanas caem abaixo de q05**;
- global h1: 15,5% abaixo de q05 vs 5% nominal; y>q95 só 2,9%.

### P3 — Perde para persistência em transição de regime (folds 0 e 3, h1–h4)
h1 fold0: 64,7 vs 59,0; fold3: 15,1 vs 10,6. h4 fold3: 41,6 vs 18,8 (perde 2,2×).
Corr h1 modelo 0,904 vs persistência 0,909.

### P4 — Piso estrutural (~50 casos) em semanas reais <10
Bins: real <10 → bias +36 (h1) a +96 (h8); real 10–50 → +20 a +62.

### P5 — Erro concentrado no top-10% de y
Top-10% (média y 483): MAE 77 (h1) / 198 (h4) / 212 (h8) vs 31/57/74 no resto.

### P6 — Timing degrada com h
Pico: ±1–2 sem em h1/h2; h4 até 6 sem; h8 até 8 sem (fold3 pico previsto em fase errada).

### P7 — Média de folds esconde o pior caso
`ensemble_avg` (MAE h1 43,6) mistura fold0 (64,7) com ano calmo (17,6).

### P8 — Recalibração de quantis usa apenas holdout 2025 (ano fraco)

### P9 — Cov50 cai a 0,40–0,41 em h7–h8 (nominal 50%)

## Notas positivas
- Bate o baseline sazonal em todos os folds/horizontes (h1: 64,7 vs 213; 15,1 vs 112).
- Sem crossing de quantis nas previsões salvas.
- Timing h1/h2 do pico bom (±1–2 semanas).

## Recomendações (implementadas na sequência deste relatório)

1. **CQR** (conformized quantile regression) por horizonte × regime-base, calibrada
   nos 4 folds com pesos temporais decrescentes (em vez do holdout único 2025).
2. **Features de regime da temporada corrente**: razão YTD vs ano anterior, tendência
   log-razão 3–4 semanas, proxy de R efetiva (casos / média 4 semanas anteriores).
3. **Correção multiplicativa de tendência** (quebra saturação de árvores nos picos).
4. **Gate de implantação por fold** no retreino (worst-fold MAE vs baseline).
5. Ajuste de amostragem/potência de loss para enfatizar semanas de alta magnitude.

— Gerado em out/2026 a partir das artefatos de validação do projeto.
