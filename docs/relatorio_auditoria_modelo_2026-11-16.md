# Auditoria técnica do Episense

Data de referência: 16/11/2026, Campo Grande. Escopo: previsão semanal de dengue, da entrada dos dados à publicação e avaliação. Esta revisão complementa o [histórico de correções](relatorio_correcoes_2026-11-16.md). Os achados abaixo distinguem defeitos confirmados, diferenças metodológicas e hipóteses a testar. Prioridade P0 bloqueia uma alegação confiável de desempenho. P1 afeta utilidade ou operação. P2 depende de evidência para justificar maior complexidade.

## 1. Decisão técnica

**Corrigir primeiro a identidade temporal de cada previsão.** A mesma coluna é usada como origem no treino e como alvo em partes do dashboard, da calibração e das métricas. Isso pode deslocar o histórico em até oito semanas e alterar quais dados definem o regime de calibração. O próximo benchmark deve ser gerado do zero a partir de um contrato temporal verificável. Os valores atuais servem como diagnóstico legado.

O esquema de publicação atômica e o replay das novas emissões já existem. O replay verifica se a previsão arquivada pode ser reproduzida. O walk-forward estima como determinada política teria se comportado em temporadas passadas. Ambos são úteis e medem coisas diferentes.

## 2. P0 | Calendário, alvo e chaves do dataset

**Erro confirmado: `test_se` é a origem.** O alvo é criado com `shift(-h)` em [data/features.py](../data/features.py#L850). O treino salva o valor deslocado em `y_test_casos` e conserva a semana da linha de entrada em `test_se`. [scripts/train.py](../scripts/train.py#L1436)

| Horizonte | Registros verificados | Desfecho igual à contagem na semana `test_se` | Desfecho igual à contagem na semana `test_se + h` |
|---|---:|---:|---:|
| H1 | 209 | 5 | 209 |
| H4 | 209 | 2 | 209 |
| H8 | 209 | 4 | 209 |

Consequências a corrigir juntas:

- O gráfico histórico indexa o ponto pelas semanas salvas e o exibe sem deslocar para o alvo. O painel por origem faz o inverso, procura `origem + h` num índice de origens e pode exibir a previsão de uma emissão posterior. [dashboard/app.py](../dashboard/app.py#L1025), [consulta por origem](../dashboard/app.py#L1277)
- O treino agrupa o alvo deslocado pela semana da origem no baseline. O recálculo usa uma série semanal de casos e recebe as semanas de origem nos blocos. Isso produz referências diferentes e exige reavaliar o ganho sazonal. [Treino](../scripts/train.py#L669), [recálculo](../scripts/recompute_metrics.py#L278)
- Os consumidores que recebem uma semana assumidamente alvo e subtraem o horizonte podem buscar o regime em `origem - h`. O limite sazonal superior também deve usar o mês do alvo correto. Conferir o contrato de cada consumidor antes de reaproveitar a calibração. [Calibração](../models/calibration.py), [recálculo](../scripts/recompute_metrics.py#L578)

**Correção.** Criar um registro versionado com `origin_se`, `target_se`, `horizon`, `issued_at`, `data_as_of`, `fold_id`, versão do modelo, quantis e procedência do desfecho. Migrar o artefato legado somente após verificar desfechos para todos os horizontes. Centralizar a soma de semanas no calendário epidemiológico e construir o alvo por calendário, validando grade completa antes de usar deslocamento por linha.

**Aceite.** H1 a H8 devem respeitar `target_se = add_epiweeks(origin_se, h)`. Usar exemplos com virada de ano, semana 53, semana ausente e limite de fold. Gráfico e cartões devem apontar para a mesma previsão. Conferir também mês do alvo, referência sazonal e momento do regime de base.

## 3. P0 | Walk-forward e vazamento temporal

| Ponto | Diagnóstico | Correção e verificação |
|---|---|---|
| Separação no tempo | O treino já separa validação interna de early stopping e usa gap nominal de oito semanas. O gap evita interseção trivial de alvos em grade regular, mas não representa a disponibilidade real de rótulos e revisões | Para cada fold e emissão, registrar o último alvo conhecido e a data em que se tornou disponível. Aplicar embargo para horizonte e atraso de notificação. Validar desigualdades de data, independentemente do número de linhas |
| Protocolo de emissão | O backtest atual ajusta um modelo por fold e percorre origens semanais. A produção pode retreinar por mudança na base ou por nova semana | Comparar política congelada por temporada com a cadência realmente contratada. Definir horário de emissão e frequência de recalibração antes de avaliar |
| Revisão retrospectiva | O treino usa base consolidada. A inferência pode substituir casos recentes por estimativas de atraso e receber outro recorte de clima | Reconstituir entradas pela data de coleta e registrar disponibilidade por fonte. Avaliar uma trilha consolidada, rotulada como retrospectiva, e uma trilha operacional com vintages verificáveis |
| Pool de calibração | O fluxo expanding filtra por número de fold. Com alvos deslocados, linhas do final de um fold anterior podem ainda não estar maduras na primeira emissão do seguinte | Filtrar pelo timestamp de disponibilidade do desfecho, além da ordem do fold. Usar conjunto de treino, reserva de calibração e teste externo distintos |
| Seleção após ver os resultados | Perfil de bandas, fator 1,08, limites e vários experimentos foram escolhidos com conhecimento do histórico. Tratar como risco de sobreajuste de seleção | Congelar candidatos e critérios antes do teste externo. Usar origem expansiva interna para ajustes e reserva externa ou emissão prospectiva para a estimativa final |
| Engenharia de atributos | Médias móveis incluem a própria linha. Isso é legítimo quando essa contagem já estava disponível | Testar invariância a modificações depois do corte. Catalogar cada atributo com fonte, data de disponibilidade, janela e grupo. Verificar especialmente dados revisados, variáveis de origem externa e tratamentos globais |
| Folds e agregação | Os artefatos medidos têm 52, 52, 53 e 52 origens. O nome “2022–2025” se refere à entrada das temporadas. O último fold termina na origem 202635 e seus alvos avançam | Informar datas de origem e alvo por horizonte. Exibir número de pares e cobertura sazonal. Classificar fold parcial pela cobertura planejada, incluindo anos de 53 semanas |

Detalhes de implementação para conferir na migração: ordenação da série antes de obter índices de split, duplicatas, separação de early stopping, janela de treino e feature selection estritamente interna. O teste de vazamento deve trocar valores futuros por sentinelas e comprovar que features e parâmetros anteriores ao corte ficam iguais. [Treino](../scripts/train.py#L173), [separação interna](../scripts/train.py#L892)

## 4. P0 | Métricas e referências

- **Definir uma biblioteca canônica por linha de previsão.** Hoje o treino e o recálculo reconstroem referências separadamente. O validador de bandas soma pinball, a biblioteca usa média e o termo WIS aparece nos dois. Usar WIS na escala padrão (duas vezes a média de pinball com cinco quantis simétricos completos) ou declarar a escala customizada. Renomear resultados de pares como pontuação de intervalo ou pinball do par, conforme a fórmula adotada. Comparar com exemplos analíticos. [Biblioteca](../models/metrics.py#L114), [validador](../scripts/validate_final_pipeline.py#L50), [Bracher et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008618)
- **Fixar elegibilidade antes de pontuar.** Atualmente máscaras finitas por quantil e por métrica permitem universos diferentes e até WIS com quantis ausentes. Usar a mesma máscara entre modelo e referência em toda comparação. Mostrar `n_eligible`, `n_scored` e motivo de exclusão. Tratar ausência de mediana como resultado inválido, impedir largura negativa e registrar abstenções.
- **Separar agregações.** Publicar métricas agrupadas no conjunto de linhas elegíveis e, quando útil, média por temporada rotulada. RMSE agregado usa raiz da média dos erros quadráticos, e WAPE usa soma dos erros absolutos dividida pelo total observado. A média de RMSEs ou WAPEs por fold mede outra quantidade. Calcular ganho como `1 − erro_modelo / erro_referência` no mesmo conjunto. Divisor zero deve resultar em indisponível.
- **Usar referências distintas.** Persistência com último valor publicado na data de emissão, média recente, sazonal do ano anterior e climatologia de semanas equivalentes. Para WIS, usar baseline com intervalos construídos exclusivamente no treino. Persistência pontual deve ser comparada por MAE. Mostrar por horizonte, temporada, regime de notificação, faixa de casos e períodos de surto.
- **Separar desfechos.** Contagem observada na emissão, estimativa de atraso e contagem madura devem ter campos diferentes. Formalizar a regra de congelamento e verificar o comportamento de revisões após 16 semanas antes de adotar esse prazo como padrão de produto.
- **Unidade de decisão.** Além de MAE, viés, WIS, cobertura e largura, medir subestimação no topo da série, antecedência e falsos alertas segundo definição acordada. Reportar tamanho amostral e incerteza por temporada. Evitar classificar cobertura como acurácia.

**Aceite.** Um conjunto pequeno de registros com casos nulos, quantil faltante, zeros e intervalo invertido deve produzir resultados esperados calculados à mão. Treino, CLI, API e dashboard devem apresentar o mesmo relatório identificado por versão. Recalcular relatórios legados e guardar os anteriores como referência histórica.

## 5. P0/P1 | Quantis, intervalos e paridade

| Achado | Correção proposta |
|---|---|
| A cadeia publicada aplica correção de nível, perfil de banda, calibração, limite de q95 e ordenação. O recompute usa sua própria cadeia. O validador de bandas pula o limite de q95 e deixa a mediana sem a amplificação usada na publicação | Extrair transformação única, parametrizada pela versão e pelo corte temporal. Usar no motor e no walk-forward. Comparar a mesma entrada nos três caminhos, antes e depois de arredondar. [Inferência](../models/inference.py#L567), [validador](../scripts/validate_final_pipeline.py#L105) |
| `np.sort` pode trocar o valor ocupando q50 após cruzamento. O perfil pode substituir caudas de modelos treinados separadamente | Medir cruzamentos por etapa e diferença em q50. Comparar ordenação, reparo com âncora de mediana e quantis treinados por largura. Declarar a regra escolhida antes da avaliação |
| Caps e pisos geram bandas mais regulares, mas podem suprimir a cauda durante surto ou gerar cobertura ruim na base próxima de zero | Testar sem limite, com limite e com regras condicionadas; medir cobertura e largura, fração abaixo de q05/acima de q95, WIS e erro em picos. Cortar comportamento visual só quando o efeito estatístico estiver demonstrado |
| Deltas aprendidos nos mesmos períodos podem produzir cobertura aparente otimista. A dependência temporal torna arriscado interpretar 90% como garantia | Usar calibração anterior à emissão, registrar tamanho amostral por horizonte e regime e mostrar cobertura empírica fora da amostra. Validar fallback para regimes raros |
| API pode preencher quantil ausente com outro quantil ou mediana | Preservar indisponibilidade explicitamente ou rejeitar previsão incompleta na rota que exige cinco quantis. Exibir intervalo ausente como lacuna. [api/main.py](../api/main.py#L939) |

## 6. P1 | Treinamento, modelos e promoção

- **Referência simples antes de complexidade adicional.** Após corrigir o contrato temporal, comparar persistência, sazonal e LightGBM atual com uma configuração mais simples. Fazer ablações de clima, lags, recursos de regime, correção de viés, perfil e calibração. Medir ganho por horizonte e custo em CPU, memória e duração. O sistema carrega 200 modelos. Ainda falta provar o ganho incremental de cada conjunto de seeds e modelos de cauda.
- **Padronizar artefatos.** Incluir hashes da entrada, código, configuração, dependências e modelos, datas de treino, folds, parâmetros escolhidos e referência da execução. O manifesto atual tem hash da base e semana treinada. O treino e o recálculo ainda leem a configuração separadamente. Congelar a configuração antes de iniciar. [scripts/retrain_pipeline.py](../scripts/retrain_pipeline.py#L312)
- **Fortalecer o gate.** O critério atual olha MAE contra referência sazonal e admite até 1,35 vezes esse erro. Estrutura completa e carga dos 200 modelos são checagens úteis. Acrescentar janela mínima madura, métricas da cadeia publicada, ganho contra persistência, cobertura/largura, desempenho em picos e paridade. Definir critérios com base nos custos do caso de uso. Testar rejeição por dados faltantes, calibração inválida e métricas inválidas.
- **Controlar ciclo operacional.** Separar execução de coleta e treino de requisições de leitura, manter um agendador responsável, testar retreino completo no ambiente Docker e simular falha/reversão sob concorrência. Validar permissões do volume, versão anterior disponível e retenção dos modelos referenciados pelo replay.
- **Expansão.** Introduzir chave município × doença × semana somente após fixar os contratos anteriores. Modelos agrupados, contagens de doenças raras e validação fora do município de treino são hipóteses de produto. Evitar aplicar limites absolutos de Campo Grande a séries de porte diferente.

## 7. P1 | Dados, emissão e auditoria

- **Ausência versus zero.** A junção semanal preenche contagem faltante com zero em caminhos do sistema. Preservar `missing`, atraso, fonte e imputação. Excluir desfechos imputados e comparar técnicas de preenchimento de features em backtest. Auditar semanas climáticas incompletas e revisões mesmo sem avançar a última semana.
- **Vintages.** Arquivos datados permitem recuperar 22 semanas com casos e clima completo até o corte desta revisão. Não há cobertura histórica de entradas para as temporadas de teste. Registrar snapshots append-only com horário real, timezone, versão do provedor, hashes e estado da coleta. Arquivos apenas com data não resolvem a ordem intradiária. Definir frequência de captura para tornar o desfecho maduro recuperável.
- **Emissão regular.** Atualmente a auditoria nasce de chamada ao dashboard. Isso tende a selecionar semanas com acesso humano. Agendar publicação, registrar falhas e dar o mesmo identificador de emissão à API e ao dashboard. Guardar o timestamp em relação ao início do alvo, pois H1 emitido após a abertura da semana-alvo tem natureza de nowcast. Reportar antecedência real em dias.
- **Replay auditável.** O fluxo atual reproduz previsões com entrada e hash disponíveis. Ainda é necessário preservar a versão do código e da dependência usada em cada replay e verificar os hashes antes de pontuar. Testar captura e replay no container, inclusive mudança de versão e retomada após falha na escrita. O relatório executado nesta auditoria apresentou zero emissões elegíveis e 27 registros sem hash de modelo.

## 8. P1 | Dashboard e bugs visuais

| Evidência no código | Efeito esperado e correção | Verificação |
|---|---|---|
| `loadHistory` devolve imediatamente se outra carga estiver em curso e aplica o resultado usando estado global | Uma mudança rápida de horizonte pode ser descartada ou mostrar dados do anterior sob o rótulo novo. Vincular resposta a horizonte, filtro e ID, cancelar ou ignorar respostas antigas e despachar a última intenção | Atrasar e inverter respostas H1/H8 em teste de navegador |
| `loadForecast` altera `state.forecast` e `state.originSE` antes de `setOrigin` conferir se a resposta ainda é a mais recente | A resposta antiga pode mudar estado e voltar a aparecer em um render posterior. Fazer a função de fetch retornar dados e aplicar somente na requisição atual | Clicar duas origens em sequência e devolver a antiga por último |
| Filtro sem linhas substitui o intervalo por 20 pontos recentes e as mensagens têm corte textual em 2026 | A tela pode mostrar dados fora do período selecionado. Exibir vazio real e derivar datas dos dados | Filtro vazio, futura temporada e mudança de ano |
| Linha de resultados recentes usa casos alterados por nowcast no campo `real`; os dois conjuntos de bandas compartilham tooltip complexo | Diferenciar observação, nowcast e alvo maduro. Mostrar história bruta, legado, emissão oficial e intervalo futuro com procedência explícita. Usar labels “intervalo preditivo” para 50% e 90% | Semana com revisão, quantil ausente, caso observado zero e pontos sobrepostos |
| Cache de previsões no navegador é por origem | Uma promoção ou revisão da mesma semana pode manter o fan antigo. Incluir identificador da emissão e garantir atualização coerente de gráfico, cards e métricas | Promover artefato com mesma semana e verificar propagação sem recarregar a página |
| Gráfico e tabela exibem métricas de artefatos distintos da previsão corrente | Integrar versão, universo, datas e rótulo de dados consolidados ou de teste prospectivo ao contexto do painel | Conferir versão e denominadores entre JSON e UI em desktop e tela pequena |

Referências: [histórico e filtros](../dashboard/static/index.html#L590), [consulta do fan](../dashboard/static/index.html#L642), [troca de origem](../dashboard/static/index.html#L730), [observados recentes](../dashboard/app.py#L1059). Os problemas de concorrência de interface foram identificados por leitura. A reprodução visual em navegador fica como teste de aceite.

## 9. Diagnóstico numérico do artefato atual

Estes números foram lidos de `models/metrics_dataframe.csv`. São referências do artefato legado e devem ser substituídos após as correções de alinhamento, comparabilidade e paridade.

| Horizonte | MAE em casos | Cobertura do intervalo 50% | Cobertura do intervalo 90% | Largura média do intervalo 90% |
|---|---:|---:|---:|---:|
| H1 | 38,60 | 52,57% | 86,09% | 245,89 |
| H4 | 65,47 | 54,43% | 88,46% | 352,44 |
| H8 | 90,12 | 34,81% | 88,46% | 520,64 |

O intervalo interno de H8 cobre menos do que o nominal no artefato, enquanto o externo é muito largo em média. Isso justifica investigar a distribuição dos erros e a utilidade de H8 antes de ajustar o visual. O ganho real sobre persistência ainda é indeterminado.

## 10. Ordem de implementação

| Ordem | Entrega | Evidência de conclusão |
|---|---|---|
| 1 | Contrato origem/alvo e correção dos consumidores | Testes de 53 semanas e histórico/cartão/arquivo coerentes para H1, H4 e H8. Validar todos os registros H1–H8 |
| 2 | Walk-forward com datas de disponibilidade, previsão por fold e cadeia compartilhada | Manifesto dos folds e teste de sentinelas. Mesma previsão pós-processada para a mesma entrada e configuração |
| 3 | Métricas, baselines e gate | Comparação pareada, agregação verificável, resultados por temporada, rejeição de artefato incompleto e reprodução |
| 4 | Emissão agendada, captura de fontes e separação no painel | Registro antes do alvo, revisão rastreável, replay em Docker, nenhum ponto associado à versão errada sob interação rápida |
| 5 | Comparação de modelos e expansão | Ablations e validação territorial após os itens anteriores. Escolher o menor sistema que preserve a utilidade operacional |

Validação desta auditoria: inspeção dos fluxos de treino, inferência, calibração, publicação e UI, confronto de 627 desfechos salvos com a base local e execução do avaliador de emissões até 16/11/2026. Nenhum treino ou deploy foi executado para este relatório.

## Implementação iniciada | Etapa 1

O treino passou a gravar `origin_se` e `target_se`, além de manter `test_se` como alias de origem para leitura pelos experimentos existentes. A grade de treino agora exige semanas consecutivas e sem duplicação. A leitura dos artefatos antigos calcula o alvo com o calendário epidemiológico. O gráfico indexa seus pontos pelo alvo, o cartão consulta a origem, a seleção de regime usa os casos da origem e a referência sazonal usa a semana-alvo. A regra opcional de limite superior consulta o mês do alvo.

A conferência com a base local encontrou 1.672 desfechos alinhados nos oito horizontes. Foram verificados a semana 53, a passagem a 202601, a rejeição de semana inválida e a paridade entre o gráfico e o cartão por origem. A geração de métricas e calibração executou em diretório temporário. Os artefatos em serviço continuam sendo os da versão atual. A próxima etapa é incorporar a disponibilidade real dos desfechos ao walk-forward e compartilhar o pós-processamento da inferência.

## Etapa 2 | Disponibilidade do desfecho e paridade com a publicação

O treino agora usa a data de corte da primeira origem de cada partição e filtra os desfechos pela semana epidemiológica em que se tornam elegíveis. A regra é configurável. O padrão é o fim da semana-alvo acrescido de 16 semanas. O embargo vale para seleção de atributos, treino interno, janela de validação usada pelo fold e treino final. Os blocos novos registram as datas de corte e de disponibilidade para verificação.

A inferência e o novo avaliador chamam a mesma transformação de quantis, incluindo ajuste de nível, perfil de banda, calibração, regras para a cauda e arredondamento de publicação. Para cada fold, a calibração usa apenas folds anteriores e alvos elegíveis no primeiro corte do fold. O fluxo de publicação gera `walk_forward_evaluation.json` no candidato antes da troca de versão. Os indicadores legados continuam sendo calculados separadamente até a padronização da próxima etapa.

Para inspecionar a avaliação sem gravar artefatos:

```sh
.venv/bin/python scripts/evaluate_walk_forward.py
```

Na prévia com 1.672 previsões históricas, 1.580 passaram pelo corte para pontuação. Os artefatos antigos não registram o novo embargo do treino e o relatório indica essa limitação. As contagens históricas são consolidadas. A regra de maturação é uma aproximação, pois o acervo disponível não permite reconstituir a data real de disponibilidade de cada desfecho. O perfil de bandas e o ajuste de nível continuam fixados pela configuração atual. Para medir generalização sem efeito dessa escolha, ainda é necessária uma reserva externa ou uma avaliação prospectiva.

A verificação do treino reduzido (uma seed, H1 e H8) confirmou os cortes dos quatro folds em cada horizonte e pontuou 395 previsões. Alterar desfechos futuros para valores sentinela deixou as calibrações anteriores iguais. O motor atualizado reproduziu os quantis publicados pela versão ativa na origem 202644. API e dashboard foram reconstruídos e passaram pelas verificações de saúde do Docker.

## Etapa 3 | Métricas pareadas e referências

O painel passou a priorizar o relatório walk-forward de esquema 2. A tabela informa o número de pares e o MAE do modelo, da persistência e da climatologia sazonal. A API expõe o relatório integral em `GET /metrics/walk-forward`. A rota legada `GET /metrics` mantém o contrato anterior. O relatório da versão atual foi calculado sem recalibrar ou trocar os modelos ativos.

O cálculo usa a mesma máscara para todos os métodos. Desfechos ausentes ou imputados, referências indisponíveis e previsões inválidas, incompletas ou com quantis invertidos ficam fora da pontuação. O WIS padronizado usa duas vezes a média das cinco perdas pinball. Os escores de intervalo de 50% e 90% são calculados separadamente. RMSE e WAPE agregados são calculados sobre as linhas, e ganho com referência de erro zero fica indisponível. O gate mantém seu critério legado de desempenho e também exige embargo verificável e métricas pareadas válidas em cada horizonte.

A persistência do histórico usa a contagem consolidada na origem. A referência sazonal usa quantis empíricos de semanas-alvo equivalentes, com desfechos elegíveis no corte inicial de cada fold. As 1.580 previsões elegíveis após o corte de maturação entraram na comparação. No recorte consolidado, a persistência teve MAE menor em H1 (35,69 contra 39,59) e H2 (52,13 contra 54,78). O modelo teve MAE menor de H3 a H8 e superou a climatologia nos oito horizontes. Esses resultados ainda dependem dos dados consolidados e dos artefatos históricos sem registro completo de embargo.

Foram conferidos casos analíticos de WIS, escores de intervalo, zeros, quantis inválidos, referências ausentes e agregação por horizonte. A etapa seguinte é a confiabilidade da emissão agendada e da interface, incluindo respostas fora de ordem e estados de filtros vazios.

## Etapa 4 | Emissão programada e consistência do painel

O dashboard tenta emitir diariamente às 06:12 no fuso de Campo Grande. A política exige as duas fontes e a última semana epidemiológica encerrada com dados utilizáveis. Registra uma emissão por origem com timestamp, entrada, identificação do modelo e um identificador exposto em `GET /forecast-issued/{origin_se}` e no dashboard. A navegação pode calcular uma prévia quando ainda não há emissão. A avaliação prospectiva continua exigindo que a emissão anteceda o início do alvo. Dependendo do atraso da coleta, H1 pode entrar como nowcast e ficar fora dessa comparação.

O histórico passou a preservar contagens observadas e ajustes por atraso em campos distintos. O painel descarta respostas de histórico e fan que já não correspondem à última seleção. Filtros vazios mostram um estado vazio que preserva o canvas para a próxima consulta. O limite do seletor de anos deriva do ano corrente. O polling considera revisão da base, revisão do arquivo de emissões e versão de modelo para atualizar o gráfico e as métricas.

Verificações: inversão de respostas H1/H8 e de duas origens em um ambiente JavaScript isolado, filtro sem linhas e teste de falha/repetição do job com fontes simuladas e arquivo temporário. A comprovação de uma emissão com as duas fontes reais depende da primeira execução programada elegível.

## Etapa 5 | Comparação de modelos

Foram treinadas três variantes sobre a mesma cópia da base local, com quatro folds, o embargo de maturação configurado e a cadeia de publicação. O identificador da coorte foi igual nas três execuções, com 793 previsões pareadas nos horizontes H1, H2, H4 e H8. O conjunto completo foi avaliado com uma e cinco seeds. O conjunto reduzido usou uma seed e 16 atributos de contagem e sazonalidade. Os resultados descrevem esta janela retrospectiva e esta política de treino. O conjunto reduzido remove vários grupos ao mesmo tempo, de modo que a diferença não isola a contribuição do clima.

| Horizonte | Pares | Persistência MAE | Sazonal MAE | 16 atributos / 1 seed MAE | Completo / 1 seed MAE | Completo / 5 seeds MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 201 | 35,69 | 125,71 | 48,29 | 37,90 | 37,16 |
| H2 | 200 | 52,13 | 126,09 | 61,97 | 57,49 | 55,42 |
| H4 | 198 | 87,17 | 126,79 | 81,46 | 80,40 | 80,08 |
| H8 | 194 | 139,14 | 128,06 | 110,76 | 92,52 | 101,75 |

| Variante | Modelos (4 horizontes) | Peso dos arquivos | Tempo local | WIS H1 | WIS H2 | WIS H4 | WIS H8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 atributos / 1 seed | 20 | 5,23 MiB | 4,4 s | 32,51 | 39,95 | 52,33 | 74,87 |
| Completo / 1 seed | 20 | 10,98 MiB | 36,1 s | 28,48 | 37,27 | 50,91 | 64,57 |
| Completo / 5 seeds | 100 | 48,28 MiB | 117,2 s | 28,23 | 35,95 | 49,58 | 68,05 |

O conjunto de 16 atributos perdeu para o completo com uma seed em MAE e WIS nos quatro horizontes. Cinco seeds reduziram MAE e WIS em H1, H2 e H4 em relação a uma seed. Em H8, uma seed reduziu MAE e WIS e teve menor MAE em dois dos quatro folds. A cobertura nominal de 90% em H8 ficou perto de 73% e 75% para uma e cinco seeds, respectivamente. A persistência manteve MAE menor em H1 e H2. Uma política por horizonte é uma hipótese que exige teste em emissões novas.

Os tempos são medições de cada execução local, com quatro workers. Não houve medição de latência de serviço. Os modelos ativos permaneceram na versão anterior. Para orientar a próxima rodada, testar persistência em modo paralelo em H1/H2, repetir o ensemble reduzido com outras seeds em H8 e remover apenas os atributos climáticos em uma ablação separada. Escolher um conjunto de holdout ou emissões futuras antes de decidir por promoção.

Reprodução em diretório novo, fora dos modelos ativos:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_model_variants.py --output logs/model-benchmark-nova-rodada
```

## Acompanhamento de H1/H2 e sensibilidade de H8 às seeds

Em H1 e H2, o arquivo de cada nova emissão oficial registra a última contagem observada disponível, sua semana de origem e a semana-alvo. A avaliação prospectiva separa previsões emitidas antes da semana-alvo das estimativas emitidas durante a semana. No corte de 21/11/2026 havia zero origens elegíveis, com maturação de 16 semanas.

Em H8, as duas rodadas usaram a mesma base e a mesma coorte de 194 previsões em quatro folds. Todas as variantes abaixo usam o conjunto completo de atributos. MAE e WIS menores indicam melhor desempenho.

| Seeds | MAE (casos) | WIS | Cobertura do intervalo de 90% |
|---|---:|---:|---:|
| 42 | 92.52 | 64.57 | 73.2% |
| 123 | 99.39 | 66.36 | 74.7% |
| 456 | 105.40 | 71.85 | 71.6% |
| 789 | 109.98 | 76.38 | 70.6% |
| 999 | 98.56 | 66.90 | 73.2% |
| 42, 123 | 96.07 | 64.37 | 74.2% |
| 42, 123, 456, 789, 999 | 101.75 | 68.05 | 74.7% |

A seed 42 obteve o menor MAE nesta coorte. A cobertura ficou abaixo dos 90% nominais em todas as variantes. A escolha de seed nessa mesma amostra exige confirmação em semanas independentes antes de alterar o modelo publicado.

### Validação temporal da correção dos intervalos de H8

Foi reservada a última janela do experimento de cinco seeds e quatro horizontes para um teste único da calibração. A regra usa os 133 erros de H8 de folds anteriores já maduros na SE 202537, pelo critério de maturação de 16 semanas. O escore é a distância do observado ao intervalo publicado (zero quando dentro), dividida pela raiz quadrada da mediana prevista mais um. O quantil conformal de 90% desses escores define uma margem congelada, reescalada pela mediana de cada previsão e aplicada aos dois limites externos. Os limites são arredondados para fora, com piso de zero casos.

Nas 37 origens elegíveis entre 202537 e 202620, a cobertura passou de 29/37 para 37/37, a largura média de 203,03 para 353,38 casos e o WIS de 44,07 para 45,83. O MAE da mediana permaneceu em 71,11. A maior largura e a piora do WIS desaconselham publicar esta regra. O período foi separado do ajuste desta calibração, mas já integrava o benchmark retrospectivo. As contagens são consolidadas e a disponibilidade histórica usa embargo aproximado. A validação em emissões novas continua pendente.

Reprodução a partir da raiz do projeto:

```sh
.venv/bin/python scripts/evaluate_h8_calibration.py logs/model-benchmark-20261117-000956/full_five_seeds
```

### Etapa 2 — Desempenho por fase em Campo Grande

Benchmark retrospectivo de cinco seeds. Alta corresponde a incidência na semana-alvo igual ou superior ao percentil 75 histórico (22,40 casos por 100 mil), calculado com semanas observadas já maduras no primeiro corte de treino externo. Baixa reúne as demais semanas. Subida significa mais casos no alvo que na origem. Pico é uma semana de alta com incidência estritamente superior às quatro semanas anteriores e às quatro seguintes. A identificação do pico usa o futuro exclusivamente para classificar o resultado observado. Um período de alta pode conter vários picos locais.

| Fase no alvo | H4 semanas | H4 MAE modelo / persistência | H8 semanas | H8 MAE modelo / persistência |
|---|---:|---:|---:|---:|
| Baixa | 150 | 41,07 / 40,53 | 146 | 52,16 / 81,01 |
| Alta | 48 | 202,00 / 232,92 | 48 | 252,60 / 315,98 |
| Subida entre origem e alvo | 84 | 99,43 / 100,56 | 79 | 142,32 / 166,19 |
| Pico local | 5 | 216,00 / 219,80 | 5 | 248,60 / 324,00 |

A mediana prevista indicou crescimento em 58/84 subidas em H4 e 63/79 em H8, inclusive em 3/5 e 4/5 dos picos, respectivamente. Esses acertos de direção não medem o acerto da data de máximo. Entre origens abaixo do limiar, H4 sinalizou 1/12 alvos altos e emitiu 7 sinais para 138 alvos baixos. H8 sinalizou 4/21 e emitiu 4 sinais para 125 alvos baixos. O sinal usa a mediana prevista e a população da origem. As contagens são de pares semanais e podem incluir a mesma onda diversas vezes. O limiar é descritivo e a análise usa dados consolidados.

Para reproduzir, executar `.venv/bin/python scripts/evaluate_epidemic_phases.py logs/model-benchmark-20261117-000956/full_five_seeds`. A rotina verifica se as semanas são únicas e se os MAEs agregados reproduzem o benchmark.

### Etapa 3 — Primeira emissão prospectiva pareada de H1/H2

Em 21/11/2026, a execução manual da rotina de emissão no Docker arquivou a origem 202645. O replay reproduziu os quantis da emissão. Os 27 registros legados sem hash ficaram fora dessa verificação. Para H1 (SE 202646), a mediana foi 53 casos e a emissão ocorreu durante a semana-alvo. Para H2 (SE 202647), a mediana foi 58 e a emissão antecedeu a semana-alvo em um dia. A persistência foi 1 caso observado em ambos. H1 e H2 são deslocamentos em relação à semana de origem; a data efetiva da emissão determina a antecedência operacional.

As primeiras comparações exigem contagens das semanas-alvo coletadas a partir de 14/03/2027 e 21/03/2027, respectivamente. O avaliador passou a conferir também a data de coleta do desfecho. No corte de 21/11 havia zero pares maduros. A execução automática futura depende do Docker Desktop e das fontes de dados.
