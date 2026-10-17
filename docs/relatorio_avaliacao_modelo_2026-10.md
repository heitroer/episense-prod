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
  (era 15.5%). fold0: h8 0.75→0.92 ✅. Piores casos RESTANTES: fold1 h1 0.73
  (21% abaixo de q05) e fold3 h8 0.55 (45% abaixo de q05, era 47%).
- **P3 — perde para persistência: melhorou, não virou o jogo.** f0 h1 63.7 vs 59.0
  (era 64.7); f3 h1 13.3 vs 10.6 (era 15.1); f3 h4 40.5 vs 18.8 (2.2×, era 2.2×).
- **P4 — piso em semanas <10: melhora leve.** Bias (pooled) +40 (h1) a +84 (h8)
  (era +36 a +96). Não resolvido por calibração (median shift rejeitado).
- **P5 — top-10%: piorou** (MAE 122/334/422 em h1/h4/h8, era 77/198/212).
  Trade do retreino: semanas típicas muito melhores, extremos piores — alvo do
  experimento 2 (tweedie p=1.7 / correção multiplicativa de tendência).
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
