# Expansão experimental da vigilância de arboviroses

## Pergunta e desenho

A troca de município foi testada com dengue em Campo Grande e Dourados. A troca de doença foi testada em Campo Grande com dengue e chikungunya. O controle foi dengue em Campo Grande. Em 21/11/2026 foram obtidas três séries públicas semanais do InfoDengue, com 619 registros por série e calendário comum entre as SE 201501 e 202645. Município, doença, contagens e meteorologia correspondem à série de origem. As análises são retrospectivas, em diretório de experimentos separado da publicação. [Documentação da fonte](https://info.dengue.mat.br/services/api).

A primeira avaliação usa o pipeline do EpiSense com seleção de até 22 atributos de casos, sazonalidade e clima, uma seed e quatro horizontes. Os três modelos foram treinados localmente em cada série, com a seleção, o embargo, os quantis e o pós-processamento walk-forward do projeto. O clima vem do Open-Meteo das respectivas cidades, com coordenadas aproximadas de −22,221 e −54,806 para Dourados. Os limiares e perfis do pipeline foram desenvolvidos originalmente em Campo Grande. A comparação avalia uma adaptação experimental do sistema. A versão operacional de 200 modelos não foi implantada para outro município.

A segunda avaliação compara transferência direta e adaptação local sob o mesmo algoritmo. O modelo reduzido usa LightGBM com 150 iterações, 15 folhas, seed 42, cinco quantis, temperatura e umidade municipais do InfoDengue, lags e médias móveis de incidência e sazonalidade. Na transferência, treina-se exclusivamente em dengue de Campo Grande e convertem-se as taxas previstas à população da localidade de destino conhecida na origem. Na adaptação, o treino usa somente a série do destino. Os folds iniciam nas SE 202236, 202336, 202436 e 202537. Os rótulos de treino são elegíveis apenas após o fim da semana-alvo mais 16 semanas. Só foram pontuados alvos maduros até a SE 202646. As origens são idênticas entre métodos, dentro de cada horizonte.

A contagem revisada disponível hoje pode divergir daquela que a equipe teria visto na data da emissão. Por isso, o experimento de transferência tem duas versões. Na primeira, tanto os modelos quanto a persistência recebem apenas casos com pelo menos 17 semanas de atraso. Na segunda, ambos usam a contagem consolidada na origem. O atraso extremo testa a dependência de informação recente; a versão consolidada mede uma condição retrospectiva mais favorável. Nenhuma delas reconstitui as notificações efetivamente disponíveis em cada data.

## Transferência direta e adaptação

MAE em casos para H8, com 195 origens comuns. Valores menores indicam menor erro. A cobertura se refere ao intervalo nominal de 90% do modelo local.

| Série e disponibilidade da contagem | Persistência | Transferência de dengue CG | Modelo local | Cobertura local |
|---|---:|---:|---:|---:|
| Dengue CG, consolidada na origem | 138,45 | — | 96,12 | 81% |
| Dengue Dourados, consolidada na origem | 41,97 | 32,70 | 25,76 | 74% |
| Chikungunya CG, consolidada na origem | 21,10 | 61,03 | 28,69 | 57% |
| Dengue CG, 17 semanas de atraso | 259,57 | — | 121,81 | 76% |
| Dengue Dourados, 17 semanas de atraso | 59,82 | 35,52 | 34,59 | 56% |
| Chikungunya CG, 17 semanas de atraso | 36,85 | 216,33 | 34,24 | 48% |

Em Dourados, com casos consolidados, o modelo local obteve MAE menor que a transferência direta nos quatro horizontes e melhor WIS. Com a contagem atrasada em 17 semanas, o modelo local ainda teve o menor MAE em H8, mas o modelo transferido teve melhor WIS e cobertura. A escolha operacional depende da disponibilidade de dados e do custo atribuído à incerteza.

Em chikungunya, com casos consolidados na origem, a persistência teve o menor MAE nos quatro horizontes. O histórico anual do próprio snapshot passa de 378 casos em 2024 para 5.150 em 2025. Escala variável, poucos eventos e uma mudança desse porte reduzem o valor de transportar parâmetros de dengue. O resultado sustenta desenvolver um comparador próprio para a doença antes de propor um piloto externo.

## Reprodução

Na raiz do projeto, o diretório de insumos é `logs/expansao-20261121`. Os arquivos `5002704_dengue.csv`, `5003702_dengue.csv` e `5002704_chikungunya.csv` são respostas da API pública `/api/alertcity`, com os parâmetros `geocode`, `disease`, `format=csv`, `ey_start=2015`, `ew_start=1`, `ey_end=2026`, `ew_end=45`. O SHA-256 exato de cada CSV está nos JSONs. Coletas posteriores podem conter revisões. Usar um diretório novo com os mesmos insumos para repetir o benchmark canônico.

```sh
.venv/bin/python scripts/evaluate_expansion.py logs/expansao-20261121
.venv/bin/python scripts/evaluate_expansion.py logs/expansao-20261121 --case-delay 0
.venv/bin/python scripts/benchmark_expansion_pipeline.py logs/expansao-20261121 --cg-weather data/raw/openmeteo/openmeteo_weekly_2010-01-01_2026-11-21_20261121.csv
```

As saídas `expansion_predictions_lag0.csv` e `expansion_predictions_lag17.csv` preservam os pares de origem, alvo, horizonte, método e contagem. Os respectivos `expansion_results_lag*.json` contêm MAE, WIS e cobertura por horizonte. Os resultados canônicos ficam em `canonical/<série>/local/result.json`. Zika, os demais municípios e o uso em serviço permanecem fora da evidência produzida neste teste.

## Verificação com o pipeline canônico reduzido

O segundo algoritmo da seção de transferência e o pipeline canônico reduzido têm resultados diferentes. Na adaptação local do pipeline canônico, o MAE de H8 foi 107,69 casos em dengue de Campo Grande, 32,89 em dengue de Dourados e 34,09 em chikungunya de Campo Grande, em 195 pares por série. As persistências correspondentes foram 138,45, 41,97 e 21,10 casos. As duas implementações concordam na direção da comparação. Generalizar esse resultado para Zika ou outros municípios exige testes próprios.
