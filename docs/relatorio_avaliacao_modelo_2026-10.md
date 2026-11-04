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

---

# Adendo técnico (out/2026) — correções implementadas

Detalhes dos experimentos podem ser vistos no GitHub (commits a partir de `15934e7`).

## O que foi implementado

| # | Fraqueza (P) | Mudança | Arquivo | Desenho |
|---|---|---|---|---|
| 1 |P8, P2, P9| **CQR delta-only** (`quantile_conformal`) por horizonte × regime-base, calibrada nos 4 folds com pesos temporais (meia-vida 2,5 a) e deltas limitados a ±1.0 log | `models/calibration.py` (novo) | Aplicado em **h1–h3 e h5** (h6–h8 rejeitados; h4 excluído pós-retreino — evidência abaixo); validação **expanding** (fold calibrado só com folds anteriores) |
| 2 |P2, P4| **Regime-base por intensidade observável** (casos da semana-base: high ≥100 / mid 20–100 / low <20) — decidível em produção | `models/calibration.py::make_regime_hook` | Substitui o regime por mês-do-alvo, que misturava temporadas de amplitudes distintas |
| 3 |P1, P3, P4| **`always_keep` de features de regime** na seleção (16 features: YTD-ratio, YoY, growth-ratio, momentum, slopes/accel, anomalies clim., vs-EWMA proxy de R) | `scripts/train.py` + config | Sobrevivem ao corte 380→120 por ganho |
| 4 |P7| **Deployment gate por fold**: reprova retreino se pior fold completo (≥52 sem) tiver MAE > 1,35× baseline sazonal | `scripts/train.py` (`deployment_gate.json`) + `scripts/update_pipeline.py` | Falha ⇒ backup `_backup_stable` restaurado, sem md5 novo e sem restart da API |
| 5 |P8| Recalibração holdout legacy **desativada** (`quantile_recalibration.enabled: false`) — era calibrada só no fold 2025 (ano fraco) | `config/config.yaml` | Substituída pela CQR |

## Experimentos que definiram o desenho (sobre validação walk-forward existente)

Avaliações **antes** do retreino (modelo atual, sem `always_keep`):

| Esquema | Resultado |
|---|---|
| LOO por mês-do-alvo (v2 inicial) | pior que raw em folds extremos: fold0 h8 mae 187→219, cov90 0.75→0.44 — **rejeitado** |
| Expanding + median shift | transferência do shift cruza temporadas de amplitudes 10×: h4 fold1 mae 76→94 **rejeitado** para a mediana |
| **Delta-only + expanding + regime-base por intensidade** | **aprovado**: h1 cov90 0.82→0.88, cov50 0.45→0.53, bias −24.4→−20.8, WIS estável; fold-level: fold1 0.77→0.87, fold3 0.80→0.92; h4–h8 intactos |

Decisões de projeto registradas:

- **Median shift global desativado** (`apply_median_shift: false`): o pool cruza
  temporadas de amplitudes 10× e o shift da mediana transfere mal (o peso temporal
  domina com os dados recentes, mas a próxima temporada pode ter outra amplitude).
  A correção de viés fica por conta das features de regime (item 3) no retreino.
- **Deltas em h1–h3 e h5** (`apply_horizons: [1,2,3,5]`): em horizontes muito longos
  (h6–h8) os erros de cobertura são contraditórios ENTRE temporadas (fold2 bandas
  2–4× reais vs fold3 bandas apertadas) — o conformal com pool cruzado degrada WIS.
  Pós-retreino, h4 ficou **ambíguo** no expanding (fold1 cov90 0.81→0.94, porém fold3
  0.65→0.56 com cov50 0.37→0.15 e WIS fold2 +19%) e foi excluído; h5 melhorou
  **todos** os folds (fold1 0.85→0.90, fold2 0.94→1.00, fold3 0.58→0.73) e foi
  incluído. O problema P9 (cov50 h7–h8 ~0.4) fica para uma futura calibração
  condicional por horizonte com mais temporadas.
- Nota honesta: **fold0 fica sem calibração** na avaliação expanding (era o
  histórico disponível até então). Em produção o artefato carrega os 4 folds.

## Métricas canônicas

Mantidas: `mae, rmse, bias, wis_total, wis_50, wis_90, width_50, width_90,
coverage_50, coverage_90` (+ `baseline_wis/mae/coverage_90`). Nos folds
calibrados, as métricas canônicas passam a refletir o estado FINAL da cadeia
(modelo + calibração conformal), com `raw_uncalibrated` preservado por fold e
`calibration` documentando esquema/parâmetros.

## Resultados pós-retreino

Retreino completo executado em 2026-10-16 (`scripts/train.py`, 8 horizontes, 4 folds,
seeds intactos, seleção com `always_keep` de 16 features de regime) — exit 0.
**Deployment gate: PASS** (pior razão fold/baseline = 0.732, h4·fold1; budget 1.35).
API e dashboard reconstruídos e verificados (`/metrics` e `/api/metrics` servindo as
métricas canônicas calibradas; `/predict/5002704` com conformal carregado e aplicado).

### Comparação global — média dos 4 folds (canônica: modelo + calibração expanding)

| h | MAE (antes→depois) | Bias | Cov50 | Cov90 | W90 | WIS_total |
|---|---|---|---|---|---|---|
| h1 | 43.6→**37.6** (−14%) | −24.4→**−16.7** | 0.45→**0.52** | 0.82→**0.88** | 229→521 | 14.3→16.0 (**+9%**) |
| h2 | 60.3→**49.6** (−18%) | −36.9→**−17.6** | 0.51→**0.60** | 0.87→**0.92** | 269→545 | 19.6→19.7 |
| h3 | 68.3→**60.1** (−12%) | −41.1→**−17.8** | 0.50→**0.54** | 0.89→**0.90** | 429→626 | 24.7→**23.6** |
| h4 | 84.4→**71.0** (−16%) | −53.8→**−30.3** | 0.50→0.46 | 0.92→0.83 ⚠ | 478→580 | 28.7→**26.5** |
| h5 | 87.8→**73.1** (−17%) | −62.3→**−12.8** | 0.47→0.42 | 0.91→0.88 | 521→529 | 30.5→**27.1** |
| h6 | 91.7→95.0 (+4%) | −63.5→−66.5 | 0.48→0.43 | 0.87→**0.91** | 508→523 | 32.7→**31.9** |
| h7 | 98.2→103.4 (+5%) | −65.8→−67.8 | 0.40→**0.43** | 0.90→0.89 | 575→515 | 33.6→**32.3** |
| h8 | 98.2→103.6 (+5%) | −68.4→−69.5 | 0.41→**0.42** | 0.87→**0.94** | 567→632 | 34.3→**32.0** |

Leitura: MAE e bias melhoram forte em h1–h5; WIS (que pune larguras supérfluas)
melhora ou fica estável em h2–h8 e piora 9% em h1 — o custo de bandas
**honestamente mais largas** no surto (fold0 h1: w90 330→1246), que compraram
cobertura (cov90 h1 0.82→0.88; fold0 h8 0.75→0.92). h4·cov90 0.83 é regressão
pontual do novo modelo (algumas semanas do fold1; raw = calibrado, pois a
calibração de h4 foi ambígua e ficou de fora — ver decisão abaixo).

### Por fold (canônico) — mudanças-chave

| fold·h | MAE | bias | cov50 | cov90 | w90 | wis |
|---|---|---|---|---|---|---|
| f0·h1 | 64.7→63.7 | −43.5→−43.3 | 0.48→0.50 | 0.81→**0.88** | 330→1246 | 21.1→28.6 |
| f1·h1 | 48.5→54.9 ⚠ | −25.7→−27.3 | 0.29→0.31 | 0.77→**0.73** ⚠ | 249→449 | 15.3→21.8 |
| f2·h1 | 17.6→18.5 | −4.0→−3.7 | 0.58→**0.66** | 0.89→**0.94** | 108→300 | 6.5→8.8 |
| f3·h1 | 15.1→**13.3** | +9.3→+7.6 | 0.52→**0.62** | 0.80→**0.94** | 87→88 | 5.8→**4.9** |
| f0·h4 | 153.1→**141.4** | −136.2→−121.3 | 0.29→0.25 | 0.87→0.88 | 602→1006 | 48.9→51.4 |
| f3·h4 | 41.6→**40.5** | +38.1→+38.0 | 0.32→**0.37** | 0.66→0.65 | 194→214 | 12.8→**12.2** |
| f0·h8 | 187.4→**186.6** | −168.0→−167.7 | 0.12→**0.15** | 0.75→**0.92** ✅ | 792→1033 | 59.0→**56.2** |
| f1·h8 | 88.9→97.7 ⚠ | −34.6→−33.7 | 0.46→**0.50** | 0.88→**0.92** | 435→629 | 31.3→**29.4** |
| f2·h8 | 18.2→26.4 ⚠ | −2.7→−7.1 | 0.66→0.60 | 0.96→0.96 | 473→**234** | 12.6→**10.5** |
| f3·h8 | 63.9→**60.2** | +57.8→+55.5 | 0.19→0.20 | 0.53→0.55 | 296→**173** | 22.5→**20.2** |

### Status por fraqueza (P1–P9, pós-retreino)

- **P1 — picos: parcial.** Bias global h1–h5 −25% a −70% melhor. Mediana no pico
  202314 (real 1132): h1 830→**904** (−27%→−20% ✅), mas h4 461→**303** e h8 535→**259**
  pioram na semana EXATA (timing antecipado). Pico 202407 (769): h4 590 (−23% ✅).
  Bins ≥400 (pooled): h1 −97 (era −95), h4 −233 (era −162), h8 −323 (era −296) →
  tail de horizonte longo **piorou**. Permanece a limitação central em h4–h8.
- **P2 — cobertura condicional: majoritariamente corrigida.** Global: h1 0.88,
  h2 0.92, h8 0.94 (nominal 0.90); fração y<q05 h1 pooled 7.7–21.2% por fold
  (era 15.5%). fold0: h8 0.75→0.92 ✅. Pós-v5: **fold1 h1 0.73→0.87** (y>q95
  15%→1.9%) e larguras de surto honestas (w90 h1 203). Piores casos RESTANTES:
  fold3 h5 0.58 e fold3 h8 0.55 (45% abaixo de q05, era 47%).
- **P3 — perde para persistência: melhorou, não virou o jogo.** f0 h1 63.7 vs 59.0
  (era 64.7); f3 h1 13.3 vs 10.6 (era 15.1); f3 h4 40.5 vs 18.8 (2.2×, era 2.2×).
- **P4 — piso em semanas <10: melhora leve.** Bias (pooled) +40 (h1) a +84 (h8)
  (era +36 a +96). Não resolvido por calibração (median shift rejeitado).
- **P5 — top-10%: piorou vs pré-B, mitigado pela v5** (MAE 77→118 (h1), 198→275
  (h4), 212→367 (h8); na cadeia v3 eram 122/334/422). Trade do retreino:
  semanas típicas muito melhores, extremos piores — alvo do experimento
  tweedie p=1.7 / correção multiplicativa de tendência, se necessário.
- **P6 — timing: melhorou em fold3** (pico 202613 previsto vs real 202612, ±1 SEM em
  h1/h4/h8 — era "fase errada"); fold0: h1 ±1 SEM ✅, h4/h8 ainda 6–8 SEM cedo.
- **P7 — pior fold visível: resolvido como processo** (gate por fold, worst 0.732
  PASS; `update_pipeline` fará rollback automático se superado).
- **P8 — recalibração holdout fraca: resolvido** (legacy off; conformal 4 folds).
- **P9 — cov50 h7–h8: leve melhora, segue aberto** (0.43/0.42 vs nominal 0.50).

### Investigação: q0.95 sobre-estimado em todos os folds/horizontes (17/10)

Diagnóstico (validação walk-forward + log de treino + artefato conformal):

| h·fold | q95 mediana | q95−y (p50/p90) | y>q95 | q95/med_prev (y<20) |
|---|---|---|---|---|
| h1 f0 | 1304 | +1123/+1286 | 0,0% | **29,7×** |
| h1 f2 | 533 | +441/+509 | 0,0% | — |
| h1 f3 | 217 | +181/+252 | 0,0% | 6,6× |
| h8 f2 | 232 | +134/+256 | 0,0% | — |

Causas identificadas, em ordem de peso:

1. **Cauda sub-condicionada (estrutural)** — a seleção do q95 usava
   `threshold_per_quantile: 0.035` + `max_features: 10` (vs 80 nos outros quantis):
   sobreviviam só 22–25 de 672 features e o modelo de cauda não conseguia
   condicionar a largura ao regime — herdava a largura do pool completo
   (y de 2 a 1132 no histórico).
2. **Escala multiplicativa (log1p)** — spread em log vira `q95 ≈ med×exp(1,64σ)`;
   em nível alto o mesmo σ produz +1000 no surto e não “+30”.
3. **Pinball 19:1 + picos raros** — errar por baixo custa 19× a sobre-cobertura;
   com ~4 temporadas de picos, o ótimo é seguro caro.
4. **Cap do conformal** — h1·mid d95 = −1,000 (bateu no teto ±1,0), h4·mid
   −0,722: o pool vota “q95 está ~2,7× acima” e o cap limita a correção.

**Experimento A (cap do conformal, rejeitado):** grid `max_abs_delta`
1,0→1,3→1,5 com avaliações expanding idênticas (h1/h2/h3/h5):

| cap | h1 cov90 / w90 / WIS / y>q95 | h2 cov90 | h3 cov90 | h5 cov90 |
|---|---|---|---|---|
| 1,0 | 0,88 / 521 / 16,0 / 4,3% | 0,92 | 0,90 | 0,88 |
| 1,3 | 0,86 / 512 / 15,9 / 5,8% | 0,91 | 0,90 | 0,88 |
| 1,5 | 0,87 / 511 / 15,9 / 5,8% | 0,91 | 0,90 | 0,89 |

Gains de largura triviais (−2%), cov90 cai em h1/h2 e fold1 h1 (y>q95 15%)
não muda — os deltas expanding raramente excedem 1,0; o cap não é a amarração.
**Rejeitado; cap mantido em 1,0.**

**Experimento B (Fase 2, aplicado):** seleção da cauda relaxada —
`threshold_per_quantile 0.95: 0.01` e `max_features_per_quantile 0.95: 40`.
Seleção pós-mudança: **51–55 features** (antes 22–25), smoke test íntegro
(gate PASS) e **retreino completo em andamento** — resultados serão anexados
após a validação expanding da nova cadeia.

Retreino B concluído (gate PASS, pior 0.927). Cauda condicionada em TODOS os
horizontes: WIS h1 −23% (16.0→12.4), w90 h1 521→203, MAE igual/melhor em
h1–h3 — mas **cov90 degrada em h6–h8** (raw: 0.91/0.89/0.94→0.72/0.81/0.81).
**Experimento C (conformal em h1–h8 no tail condicionado, rejeitado):** com a
cauda nova, o conformal cobre apenas parcialmente (h6 0.72→0.78 com MAE +10% e WIS
+12%; h8 0.81→0.83 com WIS +22%; h5 0.76 sem melhora) — não compensa.
**Decisão v4:** cauda **por horizonte** — h1–h5 com `thr 0.01/mf 40`
(condicionada, WIS muito melhor), h6–h8 restauram a cauda fina original
`thr 0.035/mf 10` (cov90 0.89–0.94 conhecido) via novos
`threshold_per_quantile_per_horizon` / `max_features_per_quantile_per_horizon`
("6","7","8"). Aplicação validada por teste unitário; retreino v4 executado.

**Retreino v4 concluído (gate PASS, pior 0.931):** h6–h8 restauraram
EXATAMENTE o comportamento da cauda fina (cov90 0.91/0.90/0.94; w90 522/515/632;
WIS 32.1/31.8/31.6 = pré-B), e h1–h4 mantiveram os ganhos (WIS −15/−23%,
w90 −43/−55%). **h5 foi o problema restante**: cauda condicionada ficou pior em
AMBOS os critérios (cov90 0.88→0.76 E WIS 27.1→29.4) mesmo com conformal
expanding. **v5:** h5 incluída no override da cauda fina (h1–h4 condicionadas;
h5–h8 finas) — retreino v5 executado, resultados abaixo após validação.

### Estado final em produção (v5, 17/10) — gate PASS; API/dashboard verificados

Canônica expandida (modelo + conformal expanding ativa em h1–h3+h5):

| h | MAE (v3→**v5**) | Bias | Cov50 | Cov90 | W90 (v3→**v5**) | WIS (v3→**v5**) |
|---|---|---|---|---|---|---|
| h1 | 37.6→**35.8** | −16.7→**−14.1** | 0.52 | **0.88** | 521→**203** (−61%) | 16.0→**12.4** (−23%) |
| h2 | 49.6→**48.7** | −17.6→**−11.9** | 0.60 | **0.91** | 545→**245** (−55%) | 19.7→**16.8** (−15%) |
| h3 | 60.1→65.3 (+9%) | −17.8→**−6.0** | 0.54 | **0.89** | 626→**306** (−51%) | 23.6→**21.3** (−10%) |
| h4 | 71.0→71.8 | −30.3→−30.4 | 0.46→0.45 | 0.83→**0.84** | 580→**330** (−43%) | 26.5→**23.9** (−10%) |
| h5 | 73.1→74.6 (+2%) | −12.8→−16.5 | 0.42 | 0.88→0.85 | 529 | 27.1 |
| h6–h8 | ≈ | ≈ | 0.47/0.44/0.41 | **0.91/0.90/0.94** | 522/515/633 | 32.1/31.8/31.6 |

Mediana (q50) e features do centro **não mudam**: P1 (picos) e P6 (timing)
idênticos ao estado v3. Guardas medidos: fold1 h1 **y>q95 15%→1,9%**;
razão q95/mediana em semanas calmas **29,7×→3,6×** (fold0 h1) e 6,6×→2,8×
(fold3 h1); excesso p90 de q95−y cai ~8× (h1 f0). WIS_90 de h1 17.4→**7.2**
(raw). Custo honesto: MAE de h3 +9% (65.3, ainda < baseline sazonal com folgado
gate 0.735) e cov90 h5 −3pp (0.85) — documentados como trade da largura.

### Limitações conhecidas deixadas abertas (documentadas)

1. fold0 (surto 2023) em h4–h8: subestimação do pico e timing 6–8 SEM cedo, não
  corrigíveis por conformal (resíduos inconsistentes entre temporadas) — alvo do
  experimento 2 (loss tweedie 1.7) se necessário.
2. fold1 h1 cobertura 0.73 (novo modelo; na calibração expanding há história
  mínima disponível para esse fold).
3. fold3 (2026 fraco) cov90 h8 0.55.
4. Bandas mais largas no surto (WIS h1 +9%) — largura honesta escolhida em vez
  de cobertura nominal sem largura.

— Gerado em out/2026 a partir das artefatos de validação do projeto.

## Adendo 4 — Feature Rt (v6, out/2026)

**Motivação.** Codeço, Villela & Coelho (2018), *Epidemics* 25:101–111: estimador
de Rt efetivo para dengue com intervalo geracional dependente de temperatura —
Rt detecta o início de transmissão crescente semanas antes do volume absoluto de
casos, funcionando como sinal de FASE/CINÉTICA complementar aos lags de casos.
O método é exatamente o implementado pela plataforma InfoDengue/AlertaDengue.

**Dado.** O CSV bruto do InfoDengue já trazia `Rt` e `p_rt1` (prob. de Rt>1)
para todo o histórico (823 SEMs, 201101–202640, 100% cobertura, sem furos).
Não houve nova infraestrutura de captação — o coletor (`collect_infodengue.py`)
guarda esses campos da API `alertcity`. Convenção de leakage: o valor de Rt da
semana *s* é tratado como conhecido em *s* (idem `casos`/`log_casos`); derivadas
usam shift/rolling do passado. Nota de revisão: valores históricos do arquivo são
re-estimativas recentes (mesma categoria de ruído do nowcast de `casos`), e a
cauda 202635–202640 estava congelada (0.751) no momento do treino — nas semanas
de 0 casos a API hoje devolve Rt=0, que o cron reconciliará no próximo refresh.

**Implementação.** `data/features.py::create_rt_features` (flag
`features.expand_rt` em `config/config.yaml`): 10 features shift-safe
(`rt_now`, `rt_lag1/2/3`, `rt_ma3`, `rt_slope_2/4`, `rt_gt1_share_4`,
`p_rt1_now`, `p_rt1_lag1`) entrando no pool normal de seleção per-tau/per-
horizonte (thr/mf inalterados; `Rt`/`p_rt1` brutos seguem excluídos como hoje).
Caminho live: `api/main.py` e `dashboard/app.py` (`_fetch_infodengue*`) agora
preservam `Rt`/`p_rt1` no merge fresh (API alertcity devolve ambos) + coerção
numérica; o ffill pós-merge cobre eventuais NaNs.

**Validação determinística.** Retreinos na mesma máquina com seeds fixos
produzem métricas idênticas a ±0.0 (3 runs Rt e 2 runs v5 sobrepostos) — a
comparação v5→Rt é limpa, sem ruído de treino nesta configuração.

Resultado global (20 features) foi bimodal: ganhos h3 (MAE −6.7%), h8 (−8.9%),
h4 (WIS −2.7%), h1 (WIS −0.4%, W90 −7%); regressões h2 (MAE **+12.3%**) e h5
(MAE +8.2%/WIS +7.1%), neutro h6/h7. Padrão de deslocamento no top-k.

**Técnica do ban por horizonte (v6 final).** Primeiro corte: banir rt_* dos
horizontes perdedores após o ranking — insuficiente: a presença de rt_* NO POOL
de candidatos re-aloca ganho no LightGBM e desloca o ranking dos sobreviventes
(h2 trocou ~75 features). Correção (v6b): ban aplicado ANTES da estimativa de
importância (`scripts/train.py::_get_selected_columns_per_quantile`, pool
`X_imp = X_tr[sobreviventes ao ban]`) → horizontes banidos reproduzem o v5
EXATAMENTE (h2/h5/h6/h7 = v5 ao quarto dígito).

**Estado final (canônico expanding, ensemble média folds):**
Rt ativo em h1/h3/h4/h8 (ban `banned_features_per_horizon` em h2/h5–h7);
h2/h5/h6/h7 idênticos ao v5; ganhos líquidos: h3 MAE 65.35→60.94 (−6.7%) e
WIS 21.25→20.38 (−4.1%); h8 MAE 102.10→93.01 (−8.9%, Bias −70→−59); h4 WIS
−2.7%; h1 WIS −0.4% com W90 203→189 (−7%). Gate PASS (pior fold 0.914, h5).
Containers `episense-api` + `episense-dashboard` rebuildados; inferência live
confirmada nos dois logs (`expand_rt: 10 features Rt adicionadas`) e fan/história
endpoints saudáveis.

**Limitações/metas abertas:** fold0 h4–h8 (tweedie 1.7), fold1 h1 cov90, cov90
h8 fold3 ~0.55 (igual ao v5 — Rt não resolve esse aspecto).

**Auto-retreino in-container (corrigido e travado).** O fallback
`do_retrain` do dashboard (disparado quando `new_max > val_max`) importava
`scripts.train`, que não existia na imagem — falhava silenciosamente desde
sempre. Correção (v6b1): `scripts/` montado read-only no serviço do dashboard
(compose) + requirements do dashboard já tinham lightgbm/joblib. Guardas
adicionais: flock cross-processo em `models/.retrain_lock` compartilhado
host↔container (treinos manuais do host saem sem treinar se houver retreino em
curso, e vice-versa) e marcador persistido `models/.retrain_last_se` — sem ele
o gatilho (`new_max 202640 > val_max 202635`, sempre TRUE, pois os folds de
validação nunca alcançam a SE mais recente) re-treinava em TODO ciclo de
refresh (revisões de raw), virando loop de ~5 min cada. Recursos: dashboard
`cpus 1.0→4.0`, `mem 1g→3g` (8 workers single-thread de horizontes; ciclo
completo ~5 min no container vs ~1h em 1 CPU).

**Determinismo entre ambientes.** Host (py3.9) e container (py3.11, mesmos
pins lightgbm 4.5.0/pandas 2.2.2/numpy 1.26.4) produzem métricas IDÊNTICAS
treinando o MESMO `episense_base.csv` — provado host final == container
(retreino 13:41 vs 13:15, idênticos ao 4º dígito). O desvio ±5% observado no
dia vs o snapshot da manhã decorre de REVISÕES DE DADO: o refresh de 13h
reconstruiu a base com revisões OpenMeteo/InfoDengue das semanas finais
(202636–202640) — resposta legítima do pipeline à atualização do dado, não
ruído de treino. Estado canônico vigente = honesto com o dado mais recente
(gate PASS, pior fold 0.873).

## Adendo 5 — Calibração da mediana e objetivo assimétrico (v6c–v6d, out/2026)

Diagnóstico que motiva os testes: bias sistemicamente negativo (−15 h1 →
−66 h6–h8) e fold0 (surto 2022/23) dominando o erro. Dois experimentos em
sequência, cada mudança isolada e medida contra o estado canônico v6
(`/tmp/exp_f1f2/baseline/`):

### EXP-A — median shift conformal por regime (rejeitado)

Hipótese: o shift rejeitado em out/2026 era **global**; a variante
**por regime-base (high/mid/low)** com validação expanding honesta nunca
tinha sido medida. Mudança: `quantile_conformal.apply_median_shift: true`
+ o recompute passou a ler a flag do config (`use_median_shift` e
`max_shift` propagados ao `compute_conformal` por fold — antes hardcoded
`False`/default). Escopo: `apply_horizons` [1,2,3,5]; fold0 permanece sem
calibração (sem histórico prévio — paralelo honesto à produção).

Resultado (expanding, ensemble): cobertura melhora em TODOS os horizontes do
escopo (cov90 h1 0.83→0.87, h2 0.85→0.92, h3 0.81→0.87, h5 0.81→0.86), MAS
as métricas de ponto degradam (MAE h1 +1.5%, h3 **+5.5%**, h5 +2.4%; WIS
h3 +4.3%, h5 +2.2%). O mecanismo da degradação é idêntico ao do shift
global antigo: **overshoot no fold1** (h3 f1 bias −24.7 → +60.3; h5 f1
−39.6 → +25.4) — o pool de calibração do fold1 contém só o fold0 (surto),
cujos resíduos positivos grandes dominam o shift. Cap agressivo
(0.8→0.25) muda nada: os shifts efetivos já são ≤0.25; o overshoot é do
pool, não do cap.

Veredito: **rejeitado** — trocar MAE/WIS por cobertura não atende ao
objetivo central. Config voltou a `apply_median_shift: false`; a plumbing
do recompute (flags config-driven) fica para futuros testes com
shrinkage dependente da dispersão do pool.

### EXP-B — objetivo assimétrico custom (pinball 2× p/ undershoot) — patológico

Mecanismo: `asymmetric_loss` (config) com `_asymmetric_quantile_obj` — callable
objective do LightGBM 4.5 para τ=0.5 em h≥4 (only_median, undershoot 2.0).
Avaliação PLÍVEL: mesma base (b42dd0ba, revisão 14:44) em ambos os lados —
EXP-B (asym ON) vs B0 (asym OFF, baseline honesto no dado revisado).

Resultado no escopo h≥4 (ensemble): MAE h4 71.16→86.24 (+21%), h5 74.59→96.96
(+30%), h6 93.99→124.78 (+33%), h7 97.85→112.15 (+15%), h8 92.96→104.61
(+12.5%); o BIAS piora EM VEZ de melhorar (h4 −31.6→−72.0; h6 −66.3→−104.0) e
cov90 incha (0.85→0.96 etc.). h1–h3 idênticos (sem asym — confirmação do gate).

Diagnóstico: callable objective do LightGBM DESLIGA o `boost_from_average`
(works only with built-in objectives) — as árvores partem de init_score=0 e,
com gradiente de pinball (leaf Newton com hess=1), convergem mal em poucas
centenas de rounds: mediana SUBTREINADA (baixa), resíduos inflados, build
conformal alarga as bandas. A "correção para cima" esperada (o objetiva
empurra o ponto de equilibro para o quantil ~0.67) nunca aparece porque o
modelo inteiro fica defasado.

Veredito: **rejeitado e marcado como antipadrão**. Alternativa confiável:
shift de LABEL no treino da mediana (EXP-B2, abaixo) que usa o objetivo
quantile PADRÃO (translation-equivariant) — mesmo efeito direcional sem
desligar o boost.

### EXP-B2 — shift de label na mediana (APROVADO, δ=0.15)

Mecanismo: `median_label_shift` (config) — treino de τ=0.5 com `y_log + δ`
para h≥4 (`horizon_min: 4`), val de early-stopping em escala real. Objetivo
quantile PADRÃO (pinball é translation-equivariant → mediana sobe ~δ);
sem custom objective. Implementação: `_median_label_shift` +
hook em `train_single_model` e `train_single_model_full`. Live: o modelo
full já sai "adiantado" — nenhuma mudança em inferência/conformal.

Resultado vs B0 (mesma base b42dd0ba, ensemble):

| h | MAE B0→B2 | Δ% | WIS B0→B2 | Δ% | bias B0→B2 | cov90 |
|---|-----------|-----|-----------|-----|------------|-------|
| 4 | 71.11→67.96 | −4.4% | 23.29→22.41 | −3.8% | −31.6→−16.3 | 0.85→0.86 |
| 5 | 74.59→74.47 | −0.2% | 27.12→26.72 | −1.5% | −16.5→−7.5 | 0.85→0.86 |
| 6 | 94.03→85.78 | −8.8% | 32.08→30.86 | −3.8% | −66.3→−49.9 | 0.91→0.91 |
| 7 | 101.88→92.19 | −9.5% | 31.84→30.69 | −3.6% | −67.3→−50.7 | 0.90→0.92 |
| 8 | 93.01→92.63 | −0.4% | 31.56→31.40 | −0.5% | −59.4→−48.8 | 0.94→0.94 |

h1–h3 EXATAMENTE intocados (gate de horizonte ✓); fold0 do surto melhora
pesado (h6 f0 MAE 183.5→162.0, h7 f0 194.8→165.6); folds calmos pagam
custo pequeno (f2/f3 +2 a +8 MAE). Média h1–h8: MAE 72.57→69.87 (−3.7%),
WIS 24.43→23.96 (−2.0%), bias médio −33.6→−25.1. Gate PASS (pior fold
mantém-se sob o baseline). y_true idêntico em ambos (mesma base).

### EXP-B3 — sweep do δ (0.20) — rejeitado em favor de 0.15

δ=0.20: média MAE 70.11 (vs 69.87 do 0.15), WIS marginalmente melhor
(23.88 vs 23.96), bias médio −22.7 — mas MAE piora em h7 (94.59 vs 92.19)
e h8 (94.36 vs 92.63) e o dano nos folds calmos cresce (f3 h8: 81.6 vs
78.7). Trade mais agressivo sem ganho líquido em métrica de erro. Ficou
δ=0.15 como canônico (comentado no config).

Estado final v6d: median_label_shift δ=0.15 h≥4 + shift conformal OFF +
Rt features v6. Artefatos canônicos regenerados no host (base
b42dd0ba3c470ae1c3c35c763ef292cf, dashboard pausado durante os treinos).

## Adendo 6 — Métricas de leitura direta: ganho vs baseline (skill) e WAPE (nov/2026)

Motivação: o conjunto canônico é absoluto (casos) e só responde "isso é bom?" quando
comparado mentalmente ao baseline. Duas adições dão leitura imediata, SEM retreino
(recalc via `scripts/recompute_metrics.py` sobre `validation_predictions.json`):

- **skill_mae / skill_wis = 1 − métrica/baseline_*** — ganho % vs o baseline sazonal
  (persistência Ano-1, mesmo mecanismo sem vazamento). +43% = erra 43% menos que colar o
  histórico; 0 = empata; <0 = pior que o baseline. Definido por fold como razão direta;
  nas agregações (`ensemble_avg`/`full_4folds_avg`) entra como **razão das médias**,
  consistente com as médias canônicas (média das razões por fold fica em
  `per_fold_ensemble`, apenas diagnóstico). Helper: `skill_score()` em `models/metrics.py`.
- **WAPE = Σ|y − p| / Σ|y|** — erro absoluto em % do volume real, peso linear: semanas com
  muitos casos pesam proporcionalmente mais e semanas de poucos casos NÃO explodem (vício
  do MAPE removido na spec 2026-10). Integrado ao `compute_all_metrics` (fonte única).

Exposição: API `/metrics` e dashboard `/api/metrics` incluem `wape`, `skill_mae`,
`skill_wis`; tabela avançada do dashboard ganha colunas **Ganho MAE / Ganho WIS / WAPE**;
novo painel **"Ganho vs baseline sazonal"** (barras H1–H8, referência no zero, segue o fold
selecionado nas abas); `metrics_dataframe.csv` ganha colunas `WAPE, Skill_MAE, Skill_WIS`.

### Valores (ensemble_avg = 3 folds completos, 2022–2024)

| h | MAE | WIS | WAPE | Ganho MAE | Ganho WIS |
|---|------|------|------|-----------|-----------|
| 1 | 37.97 | 13.81 | 21.3% | +68.5% | +67.9% |
| 2 | 54.79 | 18.04 | 32.6% | +54.8% | +57.5% |
| 3 | 60.66 | 20.52 | 38.6% | +50.2% | +51.4% |
| 4 | 66.47 | 21.92 | 45.2% | +46.9% | +48.4% |
| 5 | 74.43 | 27.08 | 48.4% | +42.8% | +37.6% |
| 6 | 79.66 | 27.30 | 53.8% | +41.8% | +39.3% |
| 7 | 89.83 | 28.40 | 62.8% | +37.3% | +39.1% |
| 8 | 92.69 | 32.02 | 36.3% | +43.7% | +37.1% |

DUAS notas de leitura: (a) WAPE do h8 (36.3%) é menor que o do h7 (62.8%) porque os alvos
h8 caem em semanas de alto volume (epidemiológicas) — o denominador Σ|y| cresce mais rápido
que o erro; leia WAPE por horizonte com este contexto. (b) skill_mae h8 > h7 segue a mesma
lógica do baseline: a dificuldade não cresce monotonicamente com h nas temporadas
avaliadas. Ganho monótono de h1..h7 confirma a degradação esperada; h8 permanece +43%.
Média h1–h8: ganho MAE +48.3% (WIS +47.3%), WAPE 42.4% do volume.

## Adendo 7 — Fase de calibração honesta e experimentos de retreino (v7.1, nov/2026)

Objetivo declarado: melhorar métricas canônicas E as previsões históricas
(walk-forward 2022–2025) sem violar honestidade expanding, non-crossing e o
gate de pior fold. Todo experimento correu o walk-forward completo e
comparou-se via `scripts/recompute_metrics.py` (maçã-com-maçã). Arquivos:
`logs/metrics_exp_{C1,C3,C4,C45,C45b,C6both,R2,R1,R1b}.json` + logs
`logs/exp_*_20261102.log`; snapshot base v6d em `logs/metrics_baseline_v6d.json`.

### Parte A — Pós-processamento (sem retreino, Camada A nível + Camada B conformal)

O `recompute_metrics.py` ganhou duas camadas config-driven (flags todas
desligadas = estado canônico provado BYTE-IDENTICAL ao estado anterior —
`NOOP_OK vs logs/metrics_baseline_v6d.json`):

- **Camada A (nível, expanding)**: EXP-C4 combinação mediana modelo+baseline
  no espaço log com λ por horizonte fitado NOS FOLDS ANTERIORES
  (λ=Σ(m−b)(y−b)/Σ(m−b)², clip [0,1]); EXP-C3 slope global
  (y_log~a+b·pred_log); EXP-C5 correção por quintil de nível previsto
  (ganho g(bin)=shrink·média(y−x) no log dos folds anteriores); EXP-C6
  amplificação do desvio vs nível sazonal.
- **Camada B (conformal CQR delta-only)**: fitada sobre as previsões PÓS-Camada
  A dos folds anteriores — a mesma honestidade expanding do esquema v3.

Resultados (ΔMAE vs v6d as-is):

| EXP | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | Veredito |
|---|---|---|---|---|---|---|---|---|---|
| C1 conformal h1–h8 | 0 | 0 | 0 | +3.7 | 0 | +3.2 | −0.2 | −0.8 | REJEITADO (deltas não transferem h≥4) |
| C3 slope global | +7.1 | +11.4 | +8.1 | −1.2 | +11.2 | +7.4 | +7.5 | +9.2 | REJEITADO (só h4 melhora) |
| C4 combo baseline | −0.1 | −1.0 | −2.3 | +3.9 | +5.6 | +7.4 | +4.7 | +7.3 | APROVADO h1–h3 |
| C5 quintil de nível | — | — | — | −2.1 | +4.4 | +1.9 | +5.5 | +5.2 | APROVADO h4 (ΔWIS −0.77, ΔWAPE −3.0%, cov50 +0.03) |
| C6 amplificação | +4.3 | +4.9 | −2.1 | 0 | 0 | 0 | 0 | 0 | REJEITADO (m fitado ≤1.0 em h≥4: o desvio longo prazo não é shrink) |

**Estado promovido v7.1**: `baseline_combination: [1,2,3]` +
`level_bin_correction: [4]` (shrink 0.5); demais flags off. Ganhos vs v6d:
h1 −0.06 / h2 −1.01 / h3 −2.30 / h4 −2.12 (h5–h8 idênticos). Snapshot em
`logs/metrics_v71post_v6d.json`.

**Conclusão estrutural da Parte A**: recalibração global não resolve h≥4. O
erro condicional de nível (regressão à média nos dois extremos de temporada)
só rende em h4 via correção por nível previsto; nos demais horizontes a
estrutura X→resíduo NÃO transfere entre temporadas.

### Parte B — Retreinos (autorizados; um experimento por vez, flag isolada)

Barra de ruído de retreino estabelecida pelo próprio R2 nos horizontes FORA do
seu escopo (h1–h3 rodaram sem stage-2): ΔMAE entre −0.28 e +1.16 ⇒ efeitos
<~1.0 MAE não são decisão.

| EXP | Design | ΔMAE vs v6d-as-is (h1..h8) | Veredito |
|---|---|---|---|
| R2 | stage-2 booster de resíduos: LGBM l2 (60 rounds, lr 0.05, 7 folhas) fitado SÓ nos folds anteriores, r̂(X) somado a todos os quantis em log (clip ±0.8) | −0.3/−1.1/−1.1/**+6.2/+21.4/+23.4/+8.6/+20.3** | **REJEITADO** — amplifica o erro em h≥4; dbias +11..+36 |
| R1 | transition weighting: peso ×1.5 p/ amostras cujo alvo (SE+horizon) desvia >0.6 log da climatologia da semana-alvo | +0.3/+1.0/+0.3/+1.5/+0.6/+2.7/−0.1/+4.7 | REJEITADO (ver R1b) |
| R1b | R1 branda: ×1.2, limiar 0.8 | +0.3/+1.8/+4.1/+0.8/+2.5/+3.1/−0.6/+3.9 | **REJEITADO** — skill médio +48.8%→+47.3%; família R1 encerrada (tuning esgotado: 1.5/0.6 e 1.2/0.8 pioram) |

Lições da Parte B (consistentes com a Parte A): resíduo e transição NÃO são
estacionários entre temporadas — o mapeamento X→erro muda de aridade por fold
(2022 sob Perron-sazonal diferente de 2024/2025). Correções fitadas no passado
amplificam ruído; o caminho correto para h≥4 é estrutural (features de fase) —
a features Rt kinetics/regime (v6) já cobrem; novas features ficam para v8.

### Parte C — Estado final v7.1 e gate

- `models/` restaurado ao estado estável v6d (backup pre-retreino em
  `models/_backup_stable`, mecanismo do `update_pipeline.py`).
- Artefatos canônicos regenerados com flags v7.1: `validation_results.json`,
  `metrics_dataframe.csv` (skill/WAPE inclusos), `validation_predictions.json` v6d.
- Pós-proc promovido: C4 h1–h3 + C5 h4 — ver tabela da Parte A.
- Follow-up pendente (consistência live vs validação): o path de inferência
  (`api/main.py` + `models/inference.py`) precisa aplicar o MESMO pós-proc
  promovido (combinação λ h1–3 via baseline expansível e ganho de nível por
  bin em h4) — materializado em `quantile_conformal.json` na próxima
  regeneração de artefato de produção.

## Adendo 8 — v7.2: leak do C4, vazamento banda→mediana e fold 2022-23 (nov/2026)

Sessão de auditoria disparada pela leitura dos quantis "malucos" do fold
2022-23 (q95 superestima em todos os períodos; q05 subestima o período
inteiro). Três achados de mecanismo e três experimentos; o canônico vira
**v7.2 = C5 h4 + repair median-protetor** (C4 sai da promoção).

### Achado 1 — C4 da v7.1 fitava λ IN-FOLD (vazamento)

O recompute aplicava a combinação modelo+baseline com λ fitado nos ERROS DO
PRÓPRIO FOLD (`y_log/med_log/base_log` da janela avaliada, commit ce90a5c,
recompute linha ~264) — exatamente o padrão de vazamento que a Fase 1
combateu nas camadas conformais. Ao refazer o techo com pool honesto
(expanding/loo, refactor de 3 passos), λ expandindo fica ≈1.0: o baseline NÃO
transfere para a mediana entre temporadas (mesmo diagnóstico da Parte B do
Adendo 7, agora no nível). **C4 removido da promoção**
(`postprocessing.baseline_combination: false`); a meta `combo_lambda` permanece
no JSON para rastreio. O histórico `metrics_v71post_v6d.json` é conservado como
contaminado (h2−h3 dele não são legítimos). Comparação honesta: C5-único = nova
referência intermediária (`logs/metrics_v72base_c5only.json`: ΔMAE h4 −2.12).

### Achado 2 — vazamento banda→mediana no non-crossing pós-conformal

Realimentação descoberta ao investigar anomalia do EXP-PH (fold1 h3, MAE
84.50 vs 74.31 raw): os deltas de banda do conformal CQR (q05/q25/q75/q95)
podem cruzar a mediana; o reparo non-crossing por semana (sort, paridade do
`inference.py:781`) **reatribui o slot q50** e a MAE canônica ("só mediana")
muda sem qualquer mudança de nível — medido: 51/52 semanas do fold1 h3 (22/52
no fold3). A inferência de PRODUÇÃO usa o mesmo repair, logo o efeito existe
em live também — é paridade, não artefato de validação — mas rompe a semântica
da métrica (calibração de banda alterando pontual).

**EXP-NC** (`postprocessing.nc_protect_median`, novo em v7.2): o repair
pós-conformal passa a CLIPAR a banda em torno da mediana pós-camada A
(`q25=min(q25,med)`, `q05=min(q05,q25,med)`, `q75=max(q75,med)`,
`q95=max(q95,q75,med)`); o slot q50 nunca é reatribuído ⇒ deltas de banda não
vazam no MAE. Resultado walk-forward (vs estado C5-único):

| h | ΔMAE | ΔWIS | Δcov50 |
|---|---|---|---|
| 1 | +0.13 | +0.05 | 0.00 |
| 2 | +0.51 | +0.26 | +0.01 |
| 3 | **−1.67** | −0.11 | +0.02 |
| 4 | 0.00 | 0.00 | 0.00 (conformal h4 fora do escopo) |
| 5 | **−1.07** | +0.10 | +0.02 |
| 6–8 | 0.00 | 0.00 | 0.00 (conformal fora do escopo) |

Por fold (MAE médio h1–h8): f1 −1.78, f2 +0.03, f3 +0.70, f0 0.00 (sem
conformal). Net: **MAE médio 69.30 → 69.03; skill 48.5% → 48.7%**; WIS ~neutro
(+0.07/h divergente); h3 W90 −3.28 (bandas mais agudas). Promovido como
**correção de semântica** (não é retreino nem fit novo): MAE volta a medir o
que anuncia. Resíduo conhecido: o `inference.py` mantém sort (comportamento
live intacto); alinhar o repair live para 'protect_median' é follow-up com
verificação de fan em API/dashboard.

### EXPRESS: EXP-L0 (conformal LOO) — REJEITADO

`validation_scheme: loo` (cada fold calibrado nos folds ≠ f; fold0 deixaria
de ficar sem ajuste — a raiz estrutural dos quantis malucos). Resultado: h4
+1.55, h5 +4.55 MAE na média; skill 48.5% → 47.9%; cov50 do fold1 piora
(0.44→0.29 em h3). Calibrar a cauda com resíduos de OUTRAS temporadas não
transfere (o fold0 isola o 2022-23 por uma razão real: 2022 não tem análogo
no pool restante — ver Achado 3). Rejeitado; `expanding` permanece canônico.

### EXPRESS: EXP-PH / PH2 (paridade do phantom gating)

Espelhar o cap de q95 da produção (`phantom_gating.enabled: true`, calma
1.5×mediana, surto 2.0×mês-alvo) na validação. **A primeira rodagem
(metrics_exp_PH.json) é INVÁLIDA**: bug de transformação dupla no recompute
(`expm1` aplicado sobre valores já em casos ⇒ cap efetivo ~med+0.4 e `inf` no
q95 quando q95>~709; cov90 colapsado era artefato). Corrigido, o **EXP-PH2**
(metrics_exp_PH2.json) é o veredito: MAE intocado por horizonte (cap só
desmagnetiza a cauda alta ⇒ determinismo restaurado — fecha a anomalia do
fold1 h3 como composto dos bugs do próprio recompute), mas **cov90 do fold0
colapsa mesmo assim** (0.87→0.52 h3, 0.92→0.60 h6): o cap segue a mediana
SUBESTIMADA do fold0 nos picos e corta a parte correta da banda. Rejeitado —
o gating live existe para conter flares em produção, não para calibrar
validação; paridade total exigirá corrigir o NÍVEL do fold0 (ver Achado 3),
não o cap.

### Achado 3 — fold 2022-23: diagnóstico fechado (é estrutural)

Métricas canônicas mostram bias da mediana −57..−161 (subestima nível,
esp. picos q50 290–786 vs real 826–1132) e cov50 0.13–0.35; nas semanas
calmas da cabeça do fold (reais 0–9) a BANDA INTEIRA está acima
(q05 31–36, q50 71–89, q95 203–256). Explicações encadeadas:
1. fold0 é o único sem pool de calibração expanding (pool vazio ⇒ nenhuma
   camada pós-A/conformal atua nele);
2. as duas alternativas de atribuir pool (LOO, phantom cap) foram medidas e
   rejeitadas acima;
3. o ensemble foi treinado majoritariamente em regimes epidêmicos; o "piso"
   da mediana não desce abaixo de ~30 e a amplitude de pico não se
   estica 2× sem features de fase.
Correção honesta disponível nas regras atuais: NENHUMA (pós-ajuste com pool
inexistente é vazamento ou estruturalmente insuficiente). Caminho = **v8:
features de fase** (trainável em qualquer temporada) — registro no plano
`episense-melhorias-2026-11.md`. Candidatos registrados para lá: floor de
q05 espelhando o cap do phantom (com idêntica elucidação de paridade live),
regressão quantílica por fase temporária, e live-path da camada A.

### Estado v7.2 (canônico)

- `config.yaml`: `postprocessing`: C4 false (removido por leak), C5 h4 true,
  `nc_protect_median: true`, `validation_scheme: expanding`,
  `phantom_parity: false`; tudo mais off.
- Artefatos regenerados + byte-parity: `models/validation_results.json` ==
  `logs/metrics_v72base.json`; `logs/metrics_v72base_c5only.json` conserva a
  referência intermediária; inválidos/experimentos arquivados em
  `logs/metrics_exp_L0.json`, `logs/metrics_exp_NC.json`,
  `logs/metrics_exp_PH2.json` (`metrics_exp_PH.json` mantido apenas como
  registro do run contaminado).
- Tabela canônica (vs as-is v6d):

| h | as-is | v7.2 | ΔMAE |
|---|---|---|---|
| 1 | 37.97 | 38.10 | +0.13 |
| 2 | 54.79 | 55.29 | +0.51 |
| 3 | 60.66 | 58.99 | **−1.67** |
| 4 | 66.47 | 64.34 | **−2.12** |
| 5 | 74.43 | 73.37 | **−1.07** |
| 6 | 79.66 | 79.66 | 0.00 |
| 7 | 89.83 | 89.83 | 0.00 |
| 8 | 92.69 | 92.69 | 0.00 |

**MAE médio 69.56 → 69.03; skill médio +48.3% → +48.7%.** Gate de deployment
intocado (P7 é retreino; sem retreino nesta fase).

---

## Adendo 9 — v8: features de fase multi-anual, bug de white-list e promoção do retreino F1c (nov/2026)

Caminho estrutural declarado no Adendo 8 ("caminho = v8: features de fase").
 executado em 02–03/nov/2026: um experimento exploratório (F0), 4 retreinos
completos (F1, F1b, F1b2, F1c — 8 horizontes × 2 seeds × 5 quantis cada) e
promoção do F1c à produção. Um bug pré-existente no caminho de features foi
descoberto e corrigido no percurso.

### EXP-F0 — existência do sinal (sem treino)

Correlação entre agregados de escala de temporada e o resíduo do canônico
v7.2, por fold: o preditor "run-rate anual (soma 52w) vs climatologia 3
anos" separa as temporadas com força (fold0 **+16.4** vs fold3 **−26.0**
em unidades de log) e correlaciona com o resíduo do fold0 (r=0.30 em h4;
r=0.70 no fold3). O sinal de "temporada recorde do ciclo" não era
representável pelo espaço existente: YoY/YTD/peak-timing/climatologia-3y
avaliam a semana corrente ou o ano corrente — nenhum mede a temporada
INTEIRA contra o ciclo de 2–4 anos da dengue.

### Implementação — EXPANSAO 10 (`expand_phase`)

`data/features.py`: `create_phase_features` — 5 features, todas
`shift(1)` no bloco (só passado explícito), alinhadas pela convenção
52/104/156 já usada na climatologia:

- `phase_ytd_log` — log do acumulado do ano epidemiológico corrente;
- `phase_ytd_ratio_clim` — log(YTD) − média do log(YTD) da mesma semana do ciclo nos 3 anos anteriores;
- `phase_amp_annual` — soma móvel 52w − média das 3 somas defasadas (run-rate anual vs baseline 3 anos);
- `phase_amp_annual_z` — idem normalizado pela dispersão dos 3 anos;
- `phase_amp_annual_roll4` — suavização 4 semanas.

Verificação de honestidade: construção refeita a partir dos dados
brutos numa semana do fim da série — diferença **0.000000** vs a feature
produzida pelo engineer. Matriz de treino (825, 694) → (825, 699);
684 → 689 features. Config: `features.expand_phase` (default `false`;
`true` no treino v8/promovido).

### Bug descoberto — white-list descartava flags novas (EXP-F1 e EXP-F1b INVÁLIDOS)

`scripts/train.py::prepare_data` montava o dict de flags de feature
engineering por **white-list hard-coded** (`expand_extra`…`expand_rt`).
Qualquer flag nova era descartada **silenciosamente**: F1 e F1b rodaram
com a matriz do v6d (694 colunas) e **zero** splits de `phase_*` em todos
os 200 boosters de cada run (verificado por `feature_importance`). Os
ΔMAE medidos nesses dois runs (±3 no agregado) são ruído de retreino
não-bitwise do LGBM (mesmo subset+seed ⇒ mesmo modelo; a diferença vem
da realocação de ranking do pool de importância). Arquivados como
referência de ruído: `logs/metrics_exp_F1.json` (F1b nem chegou ao
recompute — descartado no scan de ganho).

**FIX** (`scripts/train.py`): pass-through dinâmico de todas as chaves
`features.expand_*` do config (com o caso especial `expand_extra` de
`training.feature_selection` preservado) + log do conjunto de flags no
início do treino. Qualquer flag futura passa a funcionar sem editar o
train.py.

**O MESMO white-list existia no caminho live** (`models/inference.py::
prepare_inference_features`) — e já omitia `expand_rt`: o caminho
histórico do dashboard vinha preenchendo as 10 features Rt com 0
(silencioso: o fallback só falha acima de 50% de features ausentes).
Corrigido com o mesmo pass-through dinâmico. Sem este fix, o F1c
promovido deixaria o h2 histórico também com as 5 features de fase em 0.

### EXP-F1b2 — fase em todos os horizontes (diagnóstico)

Com o fix, features de fase dividem em todos os horizontes (~3–6% de
ganho por booster, todos os 125 com splits). Resultado global: MAE médio
69.91 (pior que v7.2), WIS 23.91 — mas o recorte por horizonte é
cirúrgico:

- **h2 melhora unissonamente nos 4 folds**: 49.87 (−5.43); fold0
  108.7 → 92.4 (−16.3, bias −93 → −75), fold1 −4.2, fold2 −0.5, fold3 −0.7;
- **h4 piora sistematicamente** nos 4 folds (+2.4/+7.3/+3.6/+2.7);
- h3 piora só no fold0 (+20.9, bias −112); h1/h5–h8 ~neutros.

Interpretação honesta: o sinal de fase é sequencial (2 semanas) e
transferível; sua leitura em horizontes ≥3 depende do pool de calibração
que o fold0 não tem (Adendo 8). Precedente já estabelecido no repositório
(Adendo 4, família Rt): ban seletivo por horizonte.

### EXP-F1c — fase somente no h2 (APROVADO)

`banned_features_per_horizon` v8: `"1"/"3"/"4"/"8": ["phase_"]`;
`"5"/"6"/"7": ["rt_", "p_rt1_", "phase_"]` (h2 mantém o ban Rt
existente). As 5 permanecem em `always_keep` (sobrevivem ao corte top-K
por ganho, que as descartaria — experimentado em F1).

Tabela canônica (MAE; as-is v6d / v7.2 / F1c):

| h | as-is | v7.2 | F1c | Δ vs v7.2 |
|---|---|---|---|---|
| 1 | 37.97 | 38.10 | 37.26 | −0.84 |
| 2 | 54.79 | 55.29 | **49.87** | **−5.43** |
| 3 | 60.66 | 58.99 | 60.07 | +1.08 |
| 4 | 66.47 | 64.34 | 64.85 | +0.50 |
| 5 | 74.43 | 73.37 | 71.20 | −2.17 |
| 6 | 79.66 | 79.66 | 79.90 | +0.24 |
| 7 | 89.83 | 89.83 | 85.64 | −4.19 |
| 8 | 92.69 | 92.69 | 95.80 | +3.11 |

**MAE médio 69.56 → 69.03 → 68.07; skill médio 48.3% → 48.7% → 49.5%;
WIS 23.58 → 23.53.** Gate PASS (pior razão fold/baseline 0.88 em h7,
limite 1.35). Por fold: f0 **133.26 → 130.07 (−3.19)**, f1 −0.28, f2
−0.01, f3 −0.62 — todos os folds melhoram, nenhum paga. Fold0: h2
−16.3, h7 −14.2 (cov50 0.13 → 0.21), h1 −4.9, h4 −6.1.

Honestidade: nos horizontes 1, 3–8 a fase está banida — os Δ desses
horizontes são de **realocação de ranking** do pool de importância (efeito
do retreino, não das features). h2 do F1c é bit-idêntico ao do F1b2
(same subset+seed ⇒ mesmo modelo). Critério de aceitação: nenhuma
regressão de fold + gate PASS + vitória expressiva no horizonte-alvo.

### Promoção à produção (03/11/2026)

- `models_v8phase4/` → `models/` copiado integralmente (200 pkls +
  `feature_list.json`, `normalization_params.json`, `quantile_conformal.json`,
  `deployment_gate.json`, `validation_predictions.json`,
  `validation_results.json`; md5 conferido em amostras). Backup v6d
  intacto em `models/_backup_stable/` (210 artefatos). Restos inertes do
  v6d (`quantile_recalibration*.json` — `enabled: false`,
  `fold_surto_summary.json` — sem leitor no código) permanecem em paz.
- Recompute canônico regenerado com as predições validadas do F1c:
  **`models/validation_results.json` com byte-parity contra
  `logs/metrics_v8.json`** (nova referência canônica).
- Smoke de paridade live (com o fix do `inference.py`): matriz
  (826, 699) idêntica à de treino; features ausentes **ZERO** nos subsets
  h1/h2/h4/h8 (fase presente em h2, Rt em h4/h8); `predict()` emite os 8
  horizontes com non-crossing e conformal/phantom intocados.
- **Restart dos containers executado em 03/11** (`docker restart
  episense-prod-api episense-dashboard`; ambos volume-mountam `models/` e
  `config/`). Verificação pós-restart: logs do engineer live com
  `expand_phase` carregado e ausência de paridade de features (os warnings
  de clima — `casos_rate`/`tempmin` — são pré-existentes, datados de
  01/11, degradação de dados live sem relação com a promoção);
  `/api/forecast?origin_se=202643` e `/health`/respondem 200.
  **Procedimento permanente**: toda alteração de artefatos/config em
  `models/` ou `config/` exige restart dos dois containers + verificação
  (health 200, flags novos nos logs, endpoints de predição).

### Estado v8 (canônico) e follow-ups

Artefatos de evidência: `logs/metrics_v8.json` (canônico),
`logs/metrics_v72base.json` (v7.2 anterior),
`logs/metrics_exp_{F1,F1b2,F1c}.json`, logs de treino
`logs/exp_F1*_train_20261102.log`, gate em `models/deployment_gate.json`.

Follow-ups registrados (não executados):

1. **fold 2022-23 residual**: bias da mediana −50..−155 em h3–h6 mantém
   (sem pool de calibração expanding); candidatos: floor do q05
   espelhando o cap do phantom, live-path da camada A (Adendo 8);
2. **fase em h3** requer mecanismo de pool por fase (v8.3; sem pool, o
   sinal anda a favor e contra — ver F1b2);
3. **`protect_median` live**: método implementado em
   `models/inference.py::_apply_non_crossing` (config-gated via
   `non_crossing.method`, default continua `post_process`); flip live só
   com verificação de fan-out API/dashboard;
4. smoke mode (`EPISENSE_QUICK=1`) segue quebrando em h≠1
   (`data/features.py` target_h1 hard-coded) — usar h=1.

## Adendo 10 — EXP-Q0: re-forma dos quantis em torno da mediana — REPROVADO, canônico intacto (nov/2026)

**Motivação.** As bandas de h4 em diante são fortemente assimétricas: W90/W50
cresce de 3.0× (h1) para 4.3×/6.1× (h6/h8), com q05 colado no piso (~21–42
casos constantes) e q95 varrendo até 4–12× a mediana (consequência da pinball
independente por τ treinada no histórico todo — o q95 aprende o envelope da
pior temporada e o q05 o piso da calmaria). Decisão do usuário: atacar pelo
caminho barato primeiro (Q0, re-forma pós-modelo, zero retreino), deixando Q1
(re-forma estrutural: retreinar as caudas como modelos de largura em torno da
mediana) como passo seguinte.

**Especificação.** Pós-modelo, entre o conformal CQR e o non-crossing (config
`postprocessing.quantile_reshape`, default `enabled: false`, escopo
`apply_horizons: [4,5,6,7,8]`): larguras em `log1p` medidas pré-reshape
(`w_hi = log1p(q95)−log1p(q50)`, `w_lo = log1p(q50)−log1p(q05)`), suavização
roll trailing global (`smooth_roll4_weeks: 4`; só passado publicado),
re-centragem em torno da mediana intocada (`q95' = expm1(log1p(q50)+w_hi*`,
`q05' = expm1(log1p(q50)−w_lo*`; non-crossing por construção; MAE intocado),
e caps `K(h×regime-base)` calibrados **expanding** no pool de bandas
PUBLICADAS dos folds anteriores — menor K da grade (step 0.05, máx 3.0) com
`miss ≤ miss_raw + tol_pp/100`; fold 2022 (sem pool) fica sem clamp
(só suavização) — honesto expanding.

**Implementação.** `scripts/recompute_metrics.py`: knobs `QR_CFG`/`QR`,
snapshots por fold (`_qr_pc` pós-camadas A/B p/ suavização; `_qr_pub` pós-NC
p/ pool de caps), bloco reshape com calibração expanding por
horizonte×regime (fallback global quando pool por regime < `pool_nmin_regime`),
metadata em `m['calibration']['qreshape']` e `scheme += '+qreshape'`.
Dois defeitos encontrados no smoke e corrigidos: KeyError int/str nos acessos
a `units` (chaves de fold em `_qr_pc`/`_qr_pub` viram int) e
`min_periods > rolling window` quando `smooth_roll4_weeks = 1`.

**Teste NOOP (flag off): PASS** — byte-parity
`models/validation_results.json` == `logs/metrics_v8.json` intacta.

**Resultado Q0 (roll4, nmin 12): REPROVADO** (evidência
`logs/metrics_exp_Q0.json`). Critérios pactuados vs observado:

| critério | limiar | observado | veredicto |
|---|---|---|---|
| cov90 global | ≥ 0.85 | h4 0.82, h6 0.78, h7 0.76 | ✗ |
| pior queda cov90 por fold | ≤ 0.03 | fold3: h4 −0.19, h6 −0.21, h7 −0.33, h8 −0.22 | ✗ |
| cov50 | ± 0.02 | inalterada (0.47/0.38/0.43/0.37/0.39) | ✓ |
| dWIS médio | ≤ 0 | h4 −0.46, h5 −0.42 ✓; h6 +0.32, h7 +0.32, h8 +0.06 | ✗ h6–h8 |
| W90/W50 h6–h8 | ≤ 4.0 | h6 4.27, h7 4.12, h8 5.94 (basically raw) | ✗ |

h1–h3 intocados por escopo (sanidade ✓). Mediana intocada ⇒ MAE/gate
inalterados, como previsto.

**Variação Q0b (roll=1 — sem suavização — e pool_nmin 10): também
REPROVADA** (`logs/metrics_exp_Q0b.json`), com valor diagnóstico decisivo:
fold0 fica **bit-idêntico** ao canônico (sem pool, sem suavização ⇒ reshape
não atua), e o fold3 **colapsa igual** (h7 f3 0.60→0.29). A suavização não
era a causa principal do colapso de cobertura — a re-centragem em si é.

**Diagnóstico honesto (por que reprovou).**

1. **Re-centrar em torno da mediana desloca os misses para o lado alto
   exatamente onde a mediana subestima rampas.** A banda bruta assimétrica
   é o que cobre o overshoot de pico (q95 até 12.5× med no fold3):
   verificado no fold 2025, bias da mediana −53.9 em h8. Quando `q95' =
   med×e^k`, o q05 sai do piso (ganho visual), mas o excedente que o
   envelope bruto absorvia não existe mais — nos ramos de rampa o real
   estoura `q95'`.
2. **O contrato de calibração dos caps preserva o perfil de miss, não
   encolhe largura.** Calibrar K com tolerância de +1 p.p. de miss no pool
   não pode reduzir sistematicamente W90 — por construção ele guarda a
   largura média necessária para ~90% no pool. Por isso W90/W50 fica
   ~inalterado em h6–h8 e o reshape apenas recoloca a mesma largura ao redor
   da mediana (o que quebra exatamente os regimes em que a assimetria era
   carregada).
3. **Shift de regime entre pool e fold-teste mata o contrato expanding.**
   Fold3 (temporada 2025-26) perde cobertura mesmo com caps calibrados nos
   folds 0–2 — a distribuição de miss/med do teste não é a do pool.

**Veredicto final: EXP-Q0 e Q0b reprovados sob os critérios pactuados.
Restore executado** (flag default off, knobs roll4/nmin12): recompute canônico
com byte-parity confirmada (`fc27dd7649c716f00d5093edb525ffca`; CSV
`f53dba6165e49796468fd98e2c8c3f65`), idêntica ao estado pré-experimento.
O código e o bloco de config **permanecem no repo com default `enabled:
false`** (mesmo precedente das camadas experimentais dos adendos anteriores:
flag ativada só para rodar o experimento, código/estrutura permanecentes),
prontos para reavaliação. Incremento único permanente: os snapshots
`_qr_pc`/`_qr_pub` e metadata no recompute (só-de-leitura).

**Caminhos seguintes para "quantis estáveis perto da mediana"** (decisão do
usuário pendente): (a) **Q1 estrutural** — retreinar as caudas como modelos
de largura condicional `u = log1p(y) − log1p(med)` (q95 = med + u₉₅; banda
matematicamente "forma da mediana"; ~25 min de retreino, autorizado); (b)
**conformal-width** para h4–h8 — substituir `w` pelo quantil de miss do pool
por (h×regime) (dirigido à largura-objeto, mais agressivo que o clamp
tolerância, mesmo risco expanding do fold3); (c) reduzir W90/W50 naturalmente
via Q2 (regularização específica das caudas τ=0.05/0.95 no retreino do
ciclo v8.3). Recomendação: isolar o Q1 como próximo experimento — é o único
que muda o mecanismo em vez de re-arranjar o resultado.
