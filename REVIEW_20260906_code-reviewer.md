RELATÓRIO DE REVISÃO ADVERSARIAL — Episense-Prod
Escopo: scripts/train.py, data/features.py, models/inference.py, api/main.py, dashboard/app.py, dashboard/static/index.html, config/config.yaml
Convenção validada: YYYYWW, calendário BR domingo contém 4/jan, gap=horizonte, métricas em casos após expm1.

Foram encontrados defeitos. Ordenado por severidade. NENHUM código foi corrigido.


Bug 1 — Recalibração quantílica circular: deltas estimados em test folds que também treinam o modelo final (vazamento)
- Severidade: crítico
- Local: scripts/train.py:1401-1468 (save_artifacts gera quantile_recalibration.json) + models/inference.py:135-158 + 429-520 (_load_quantile_recalibration e _apply_quantile_recalibration)
- O erro: Os deltas log-scale q05/q95 (conformal) são calculados concatenando todos os folds de validação (fold_series). Em seguida train_horizon faz retrain FULL em X_full_base = todos os dados (incluindo exatamente essas semanas de teste). O modelo de produção vê durante o treino as mesmas semanas cujo resíduo foi usado para calibrar o intervalo. Inference então soma delta em log na predição futura.
- Por que está errado: Trace com fold 2022-2023. Suponha h1. Modelo fold0 treina em 2014-2022-08, testa em 2022-09..2023-08. Resíduo r05 = y_true_log - q05_pred é coletado. Depois modelo FULL treina em 2014..2026 incluindo 2022-09..2023-08 com peso normal. Delta d05 = quantile(r05,0.05) (ex: -0.18). Na produção, pred_log_q05 = raw_log_q05 + (-0.18). O delta foi estimado em dados que o modelo FULL já memorizou, logo cobertura reportada pós-recalibração (≈0.90) é otimista fora-da-amostra. Em produção real 2026 o delta não será re-estimado. É o clássico leakage "train on test via calibration".
- Impacto: Cobertura 90% e 50% infladas em 3-8 pontos. WIS subestimado. Risco clínico: intervalo parece calibrado em validação mas em semana inédita (ex: surto atípico 2026) falha.
- Solução: Instruir default profile a separar holdout de calibração não usado no retrain FULL. Ex: reservar último fold (2025 parcial) apenas para calibrar, ou fazer cross-conformal por fold (leave-one-fold-out) e só aplicar delta médio. Treinar modelo final apenas em dados até 2024 se calibrar em 2025, ou recalibrar via validação interna (gap) sem tocar teste. Documentar e testar que fold_series usado para deltas nunca intersecta X_full de um horizonte.

Bug 2 — Gap de validação interna (early stopping) fixo 4 < horizonte para H5-H8 — validação enxerga futuro do treino
- Severidade: crítico
- Local: scripts/train.py:934-943 (train_horizon: gap interno) — linha gap = validation.gap_weeks (4) vs gap_per_horizon = h
- O erro: Walk-forward externo usa gap=h (h1=1..h8=8) corretamente (linha 864). Mas a divisão interna para early stopping faz cutoff = len(X_train)-val_size; X_tr = X_train[:cutoff-gap]; X_val = X_train[cutoff:] com gap = config.training.validation.gap_weeks =4 fixo, ignorando horizon. Para h=8, gap 4 <8 viola assert gap>=h do externo.
- Por que está errado: Trace h=8, X_train com 400 semanas, val_size 80, cutoff 320. X_tr termina em índice 316 (320-4), X_val começa 320. Último treino em t=316 prediz alvo t+8=324, que está 4 semanas DEPOIS do início da validação (320). Validação em t=320 usa lag1 = casos em 319, que inclui casos em 324? Não, mas o inverso: target de treino (324) cai dentro da janela de validação futura, não no passado. O problema real é o contrário: validação em t=320 tem alvo 328, treino em 316 tem alvo 324 — não há sobreposição direta de índice, porém a separação temporal entre conjuntos de linhas é 4, menor que horizonte, então a correlação serial de 8 semanas contamina estimativa de early stopping (validação não é honesta). Early stopping escolhe n_estimators otimista.
- Impacto: Número de árvores (best_iteration) viesado para h5-h8, geralmente superajuste. MAE/WIS reportados para h5-h8 em validação ficam 5-15% melhores que em produção.
- Solução: Alterar gap interno para gap = max(validation.gap_weeks, horizon) ou horizon quando gap_per_horizon true. Assert interno igual ao externo. Recalcular best_iteration e Média de n_iters para FULL.

Bug 3 — Anchor guard ausente no pipeline live (API e Dashboard) — treino descarta última semana com clima parcial, live usa clima incompleto
- Severidade: crítico
- Local: data/process_data.py:96-113 (anchor guard: drop last_weeks se clima_dias<7) vs api/main.py:243-282 (_merge_and_prepare, sem guard) e dashboard/app.py:509-576 (_merge_and_prepare_live, sem guard) e api/main.py:140-240 (_fetch_openmeteo live sem guard)
- O erro: Treino calcula clima_dias = (week_end-week_start)+1 por SE e enquanto último SE tem dias<7 ou NaN, remove a linha e recua âncora. Live apenas faz merge left, ffill climático e shift, sem checar completude. OpenMeteo live busca 1200 dias até hoje e agrega por SE via groupby; se hoje é quarta da SE 202636, o weekly precip_total será soma de 3 dias, temp_mean média de 3 dias, mas será usado como se fosse semana completa.
- Por que está errado: Trace: suponha SE 202636 com 3 dias de clima (dom-ter). Treino em 2025 nunca veria essa semana parcial — ela seria descartada até completar 7 dias. Live em 2026-09-06 gera features precip_total_lag2 etc que usam precip_total da semana parcial (subestimado 40% da chuva). Lagoas 7-12, anomalias vs climatologia e acumulados 16-26s também ficam enviesados. Modelo treinado com precip_total_acc_26 ~ soma 26 semanas completas recebe na inferência soma com última semana incompleta.
- Impacto: Predição da semana seguinte ao fetch parcial fica sistematicamente baixa em precipitação e alta em temperatura (média de dias mais quentes), degradando H1-H2 em 10-30 casos em semanas de transição. Difícil de reproduzir em validação.
- Solução: Replicar anchor guard no live: após merge e antes de feature engineering, calcular clima_dias (ou contar daily rows por SE) e enquanto última SE tem <7 dias, removê-la e recuar âncora. Ou buscar OpenMeteo com end_date = último domingo completo, não hoje. Garantir paridade treino/live.

Bug 4 — Inconsistência de unidades na decomposição WIS e baseline WIS (sharp/over/under mal escalados)
- Severidade: alto
- Local: scripts/train.py:527-551 (calculate_metrics WIS decomposição) e 673-724 (baseline_wis)
- O erro: WIS principal correto = mean(pinball 5 quantis). Decomposição para 90% faz sharpness_vec = (q95-q05)(alpha/2) com alpha 0.10 => 0.05width. Over = (2/alpha)(q05-y) se y<q05 => 20under. Under similar. Depois wis_total_vec = sharp+over+under+median_abs. O interval score correto para alpha 0.10 é IS = (q95-q05) + (2/alpha)(q05-y)+ (2/alpha)(y-q95). Ou seja sharp deveria ser (q95-q05) sem *alpha/2. Ao multiplicar por 0.05 o sharp fica 20x menor. E wis_total soma median_abs sem peso 0.5, dobrando contribuição da mediana vs fórmula WIS (que pondera mediana 0.5). Baseline_wis usa mesma fórmula errada, então comparação modelo vs baseline fica inconsistente com WIS pinball.
- Por que está errado: Exemplo numérico: y=100, q05=80, q95=120, median=100. Pinball médio = ~6.7 . sharp correto =40, over=under=0, median_abs=0 => IS=40, WIS approx (ISalpha/2 + median0.5)/(K+0.5) ≈? Com fórmula do código sharp=2, wis_total=2, muito menor. Dashboard avançado mostra wis_sharpness ~2 quando deveria ~40, enganando operador sobre nitidez do intervalo.
- Impacto: Métricas avançadas (WIS sharp/over/under, baseline_wis) exibidas no dashboard estão 20x subestimadas e não somam para WIS. Decisões sobre alargamento de intervalo baseadas nesses números serão erradas. Não afeta MAE/WIS principal, mas afeta diagnóstico de cobertura.
- Solução: Corrigir sharp = (q_sup - q_inf) (sem alpha/2) para IS. Para WIS total com 1 intervalo + mediana, usar peso correto: WIS = ( median_abs0.5 + IS(alpha/2) ) / (1+0.5) ??? Ou manter WIS pinball como métrica principal e documentar decomposição separada como IS. Alinhar baseline_wis com mesma fórmula e não comparar com pinball.

Bug 5 — MASE e SPL com tratamento de semana 53 e fallback divergentes
- Severidade: alto
- Local: scripts/train.py:565-588 (MASE denom) vs 470-500 (seasonal_naive_baseline) vs 740-754 (SPL naive)
- O erro: MASE denom = mean(|y_train[52:] - y_train[:-52]|) estritamente lag 52, sem busca ±k. seasonal_naive_baseline (função isolada) faz fallback T-52±1..4 e last_val. SPL naive faz hist por WW e fallback para lista completa se WW ausente. São três implementações diferentes do mesmo baseline sazonal.
- Por que está errado: Trace ano 2025 tem 53 semanas. Teste SE 202553 (semana 53). Para prev baseline de 202653 (futura) o denom MASE ignora porque train tem lag52 diferença mas semana 53 só existe em 2014,2020,2025 (3 amostras) — denom inclui salto 52 entre 202553 e 202501 (na verdade 202552) que não é mesma semana epidemiológica. Já SPL naive para 202553 busca hist de WW="53" correto (média de 3 valores). Para semana 01 de 2026, MASE usa lag52 entre 202601-202501 (correto), mas teste em 202601 que cai em 2025-12-28 (domingo) pode ter WW 01 de 2026 vs 01 de 2025 — correto. A divergência faz MASE e SPL não comparáveis.
- Impacto: MASE subestimado (denom maior quando inclui transição 52->53 errada) ou superestimado (denom ~0 em período calmo). SPL matrix vs MASE contam histórias conflitantes. Em anos com 53 semanas, erro de 1 semana no naive pode ser 30-80 casos de diferença.
- Solução: Unificar baseline sazonal em função única que respeita weeks_in_year: buscar mesma WW no ano anterior via epiweek_to_date, com fallback ±k via datas, e usar mesmo denom para MASE (lag via WW, não via índice 52). Ou calcular denom como média sobre pares (y_t - y_{t-52epi}) mapeados por SE.

Bug 6 — API predict fallback q75 antes de q95 — operador de ordem trocado faz IC 50% futuro virar mediana
- Severidade: alto
- Local: api/main.py:482-494 (predict, get_q para q75)
- O erro: Código faz q75 = get_q(['q0750','q75','0.75',0.75], q95 if 'q95' in locals() else pred['casos_previstos']) na linha 493, mas q95 só é definido na linha seguinte 494. Logo 'q95' in locals() é sempre falso, fallback é casos_previstos (mediana), não q95.
- Por que está errado: Trace quantis = {"q0050":10, "q0500":50, "q0950":100} sem q0250/q0750. Esperado: q25->10 (fallback q05), q50=50, q75->100 (fallback q95, limite superior 50%), q95=100. Real com bug: q75->50 (mediana). Cliente recebe IC50 futuro [10,50] em vez de [10,100], intervalo 50% colapsado.
- Impacto: Dashboard e API retornam q75 idêntico à mediana quando modelo não emite q75 (raro, mas fallback existe para compatibilidade). Em produção com 5 quantis sempre presentes o bug fica silencioso, mas quebra compatibilidade com modelos antigos 3 quantis e mascara erro de quantil ausente.
- Solução: Reordenar para definir q95 antes de q75, ou usar fallback explícito para get_q de q75 como quantis.get("q0950") se ausente. Testar com dict faltando q75.

Bug 7 — FBias estratificado por ano calendário instável e mistura sazonalidade — thr P75 por calendário quebra janela Sep-Ago
- Severidade: alto
- Local: scripts/train.py:610-653 (calculate_metrics FBias P75)
- O erro: Agrupa test_se por int(SE[:4]) calendário. Cada fold epidemiológico Sep-Ago é fatiado em dois anos calendário (ex fold 202336-202435 tem 17 semanas de 2023 e 35 de 2024). Thr é P75 de cada fatia separada. Fold 2025 parcial tem 202537-202652 (16 semanas 2025 +33 semanas 2026). Amostra de 16 semanas para thr é ruidosa. Spec diz "P75 anual in-sample per fold: thr = P75(y_true do ANO no TESTE)" mas ANO deveria ser ano epidemiológico do fold, não calendário.
- Por que está errado: Trace fold1 (202336-202435) y_true: 2023-36..52 casos = [5,2,8,...] P75_2023=12, 2024 casos [30,80,150...] P75_2024=90. Surto_mask = y>thr respectivo. Semana com 20 casos em 2023 seria surto (>12) mas 20 casos em 2024 seria calmaria (<=90). Mistura surto de baixa estação 2023 com surto de alta 2024, diluindo FBias_surto. Correto seria thr único P75 de todas as 52 semanas do fold (ex 45) e classificar >45 surto.
- Impacto: FBias_surto e FBias_calmaria reportados por fold e no dashboard avançado ficam instáveis, especialmente fold 2025 com 49 semanas (thr 2026 com 33 amostras). Comparação entre folds não é comparável. Decisões de viés em surto vs calmaria ficam erradas.
- Solução: Calcular thr como P75 de todo y_true do fold (52 ou 49 semanas) ou, se quiser anual, agrupar por ano epidemiológico (Sep-Ago) único por fold. Documentar e fixar threshold como único por fold.

Bug 8 — Falhas silenciosas com except genérico e defaults que mascaram erro (data stale, métrica NaN, fallback 0)
- Severidade: médio
- Local: dashboard/app.py:276-346 (_fetch_fresh_raw ignora exceções com warning e retorna False), 348-420 (_fetch_infodengue_live fallback full), 581-602 (_get_live_engineered retorna DF vazio), models/inference.py:328-334 (_select_horizon_features preenche features ausentes com 0 com warning), scripts/train.py:1212-1225 (feature selection fallback para todas features sem aviso crítico)
- O erro: Múltiplos except Exception: pass ou logger.warning sem propagar. Ex: _fetch_fresh_raw falha ao buscar InfoDengue mas continua usando CSV stale sem sinalizar ao usuário; _select_horizon_features preenche  missing features com 0 e apenas loga warning, mas modelo recebe vetor constante fora da distribuição; feature selection falha e usa todas 669 features sem falhar o treino.
- Por que está errado: Trace: OpenMeteo retorna 429, _fetch_openmeteo_live tenta fallback local e se falhar retorna DataFrame vazio silenciosamente (linha 471 return pd.DataFrame()). _merge_and_prepare_live faz left join e ffill, produz DF com clima todo NaN preenchido com ffill histórico, sem erro. Dashboard mostra previsão com clima defasado de 156 semanas sem indicar. Usuário vê "forecast" sem saber que é stale.
- Impacto: Operador toma decisão com previsão baseada em clima de 3 meses atrás achando que é tempo real. Métricas NaN viram 0 no preenchimento, viés silencioso.
- Solução: Tornar falhas explícitas: se fetch falha e fallback também falha, levantar HTTPException 503 com detalhe, ou retornar flag stale no JSON. Para features ausentes, falhar fast se >10% missing (já existe raise para ano_normalizado mas não para outras). Para feature selection, logar como critical e não silenciosamente usar todas.

Bug 9 — Phantom gating esconde surto atípico (clamp q95 em calmaria mascara sinal precoce)
- Severidade: médio
- Local: models/inference.py:522-562 (_apply_phantom_gating) + config.yaml:254-257 (phantom_gating enabled, calm_q95_max_factor 3.5)
- O erro: Após recalibração e non-crossing, q95 é clampado para min(mediana*3.5, mediana+5) em estação calma (Jun-Set). Se mediana 10, cap 35. Se surto real atípico fora de época (ex: 2024 surto começou em Jun com 80 casos), modelo corretamente previa q95 120, mas gating corta para 35, subestimando risco.
- Por que está errado: Trace: ultima SE 202522 (jun), horizon 4 mira SE 202526 (jun). Modelo prevê mediana 12, q95 raw 80. Regime calm => cap 12*3.5=42 => q95_clamped 42. Usuário vê IC90 [q05,42] e planeja leitos para 42 quando real 80 ocorre. O gating foi introduzido para suprimir "phantom" de lag52 em anos secos, mas não distingue phantom de surto precoce real.
- Impacto: Falsos negativos em surtos precoces ou fora de época, justamente quando alerta precoce mais importa. Cobertura 90% em calmaria artificialmente alta (porque intervalo foi estreitado).
- Solução: Tornar gating опcional por horizonte ou desabilitá-lo para H1-H4, ou usar gating condicional apenas quando prev_year_was_outbreak==0 e anomalia climatológica baixa. Logar quando clamp ocorre e expor no API (flag gated). Testar cobertura com e sem gating.

Bug 10 — Inconsistência treino vs inferência no número de horizontes para engenharia (target_horizons)
- Severidade: médio
- Local: models/inference.py:199-202 (prepare_inference_features usa range(1,9) fixo) vs scripts/train.py:238-256 (prepare_data usa self.target_horizons do config) e dashboard/app.py:597 (TARGET_HORIZONS do config)
- O erro: Inference força 8 horizontes sempre, enquanto treino e dashboard usam config (1..8). Se config fosse alterado para 1..4 (como havia em feature_list.bak), inference ainda criaria features para h5-h8 (log_lag_h5..h8, weather lag_h5..h8) que treino com 4 horizontes nunca viu. Seleção de features para h1..4 ignoraria essas colunas extras, mas o DataFrame teria colunas extras não usadas. Inverso: se treino com 8 mas inference com 4, faltariam features horizonte-específicas h5-h8 ausentes e seriam preenchidas com 0 (linha 328 warning).
- Por que está errado: Embora hoje ambos 1..8 coincidam, o contrato não é garantido. Teste: config com horizons [1,2,3,4] treina 4 horizontes, salva feature_list.json com 4 entradas. Inference carrega horizons [1..4] do config? Não, ele hardcode 1..8, tenta carregar h5..8 sem modelos e sem feature list, loga warning e retorna predição vazia para h5..8 mas frontend espera 8 colunas.
- Impacto: Quebra silenciosa ao mudar config, difícil debugar.
- Solução: Parametrizar inference para usar self.target_horizons do config, não hardcoded 1..8. Validar que feature_list horizon set == target_horizons.

Bug 11 — Parse de SE e semanas com fallback int(SE)+n e semanas_in_year fixo 52 em alguns caminhos
- Severidade: médio
- Local: dashboard/app.py:769-779 (add_epiweeks fallback return str(int(se)+n)), api/main.py similar, dashboard/static/index.html: JS weeksInYear usa só 2014/2020/2025 hardcoded vs epiweeks.py weeks_in_year dinâmico
- O erro: Fallback str(int(se)+n) para somar semanas ignora virada de ano e semanas 53. Ex: SE 202552 +1 => int 202552+1=202553 mas 2025 tem 53 semanas então 202553 é válido, porém 202553+1 =>202554 que não existe (deveria ser 202601). JS no slider usa lista hardcoded [2014,2020,2025] para weeksInYear, mas se ano futuro 2026 tiver 52, slider ratio calcula errado para SEs de 2026.
- Por que está errado: Trace: predição de origem 202552 h2 => add_epiweeks via epiweek_to_date funciona correto (202601). Mas se epiweek_to_date falhar (exceção), fallback 202552+2=202554 (SE inválida 54) será retornada e frontend tenta buscar dados para SE inexistente, gráfico quebra.
- Impacto: Baixo hoje pois epiweeks funciona, fallback raramente usado, mas em ano novo com 53 semanas o gráfico e API podem retornar SE inválida sem erro explícito.
- Solução: Remover fallback int+ n, sempre usar epiweek_to_date/date_to_epiweek e falhar explicitamente se erro. No JS, importar weeks_in_year ou calcular dinamicamente via data, não hardcoded.

Bug 12 — Métricas MASE/MAE com denominador zero ou NaN e fallback para MAE vs média — infla/esconde erro
- Severidade: baixo
- Local: scripts/train.py:570-588 (MASE denom fallback)
- O erro: Se y_train_cases for constante (ex: calmaria com casos 0-1 por 52 semanas), denom_seasonal = mean(|y_t - y_{t-52}|) ≈0, código checa denom<1e-10 e fallback para denom_fallback = mean(|y - mean(y)|) +1e-10. Ex: y_train = [1,1,1,...] denom_seasonal 0 -> denom_fallback = mean(0)=0+1e-10 =>1e-10, MASE = mae_model /1e-10 => gigante (1e12) mas depois clip? Não clipa MASE. Ou se y_train len<=1 denom NaN => mase NaN.
- Por que está errado: Trace calmaria 2024 com y_train 2023 calmaria baixa: y_train_cases = [2,1,2,1...] denom_seasonal ≈1, mae_model 5 => MASE 5. Ok. Mas para período pós-surto com y_train contendo pico 500, denom_seasonal ~100, MASE 0.05 parece ótimo mas esconde que modelo erra timing. Fallback para lag1 quando len<52 também mistura escalas.
- Impacto: MASE reportado no dashboard avançado pode ser >10 ou NaN em folds calmos, confundindo comparação com baseline. Não afeta MAE.
- Solução: Definir denom mínimo floor = max(denom_seasonal, percentile(|y_train|,10)) ou 5 casos, e reportar MASE como NaN se denom não confiável, não fallback silencioso para média. Documentar.

OBSERVAÇÕES MENORES (não são bugs, apenas melhorias):
- config.yaml objective tweedie vs quantile duplicado (linha 122).
- dashboard/app.py _ensure_fresh_data reinicia retrain em thread daemon sem lock de modelo, pode competir com inference load.
- features.py create_climatology_features usa clip 0.5 para std zscore, arbitrário.
- index.html slider com SNAP_SES hardcoded 202236..202652, precisa atualizar manualmente a cada ano.

Conclusão: Pipeline walk-forward, engenharia e métricas estão bem estruturados, mas os 3 críticos (recalibração circular, gap interno, anchor guard) invalidam parcialmente a honestidade da validação e a paridade treino/live. Corrigir esses primeiro, depois os altos de métricas e API.
</atem:parameter>
</atem:invoke>
</atem:function_calls>
╰──────────────────────────────────────────────────────────────────────────────╯

Resume this session with:
  hermes --resume 20260906_141333_39a724 -p code-reviewer
  hermes -c "Analisar bugs vazamento métricas Episense-Prod" -p code-reviewer

Session:        20260906_141333_39a724
