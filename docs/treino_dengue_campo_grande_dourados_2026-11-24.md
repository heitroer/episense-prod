# Previsões históricas de dengue em Campo Grande e Dourados

Avaliação retrospectiva de desenvolvimento com contagens semanais revisadas. Cada município tem 1.588 pares maduros de origem e horizonte. A comparação usa as mesmas semanas-alvo da versão de referência em cada município.

## Resultados

Cada célula mostra referência → candidato. MAE, WIS, viés e largura estão em casos. Cobertura é a fração de semanas dentro do intervalo, com extremos inclusivos. O viés positivo indica superestimação.

O MAE caiu em 15 dos 16 pares de município e horizonte. O WIS caiu em 14. As semanas e temporadas usadas na seleção fazem parte desta avaliação de desenvolvimento.

Em Dourados, H4, o MAE subiu de 17,23 para 17,81 casos.

### Campo Grande

| Horizonte | n | MAE | WIS | Viés | Cob. 50% | Cob. 90% | Largura 90% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Todos** | 1588 | 80,5 → 60,7 | 53,7 → 39,3 | -14,7 → -0,7 | 48,1 → 47,7 | 81,4 → 91,6 | 262,0 → 343,5 |
| H1 | 202 | 38,1 → 32,2 | 26,3 → 20,8 | -18,1 → -1,2 | 52,0 → 47,5 | 82,2 → 89,6 | 185,8 → 150,1 |
| H2 | 201 | 46,8 → 40,9 | 32,5 → 27,7 | -8,7 → 0,2 | 61,7 → 52,7 | 86,6 → 91,5 | 229,0 → 224,1 |
| H3 | 200 | 87,6 → 51,4 | 51,9 → 34,5 | 3,2 → -2,9 | 50,0 → 47,0 | 85,5 → 91,0 | 253,4 → 299,5 |
| H4 | 199 | 75,0 → 58,7 | 46,8 → 40,4 | -15,5 → 8,8 | 52,3 → 48,2 | 86,9 → 91,5 | 286,0 → 374,4 |
| H5 | 198 | 95,1 → 67,1 | 67,7 → 43,3 | -6,9 → -0,0 | 42,9 → 47,5 | 83,8 → 92,9 | 326,8 → 392,5 |
| H6 | 197 | 95,0 → 73,7 | 65,6 → 46,6 | -29,8 → -2,1 | 45,2 → 46,7 | 79,2 → 91,9 | 261,6 → 411,6 |
| H7 | 196 | 103,5 → 80,0 | 71,0 → 50,1 | -29,8 → -5,6 | 41,3 → 45,4 | 71,4 → 91,3 | 262,1 → 432,8 |
| H8 | 195 | 104,7 → 83,1 | 68,9 → 52,2 | -12,1 → -3,3 | 39,0 → 46,7 | 75,4 → 92,8 | 293,8 → 471,9 |

### Dourados

| Horizonte | n | MAE | WIS | Viés | Cob. 50% | Cob. 90% | Largura 90% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Todos** | 1588 | 19,9 → 18,0 | 13,8 → 12,1 | -7,8 → -4,0 | 45,3 → 50,1 | 74,2 → 89,2 | 59,8 → 103,4 |
| H1 | 202 | 13,3 → 12,3 | 9,0 → 8,0 | -5,3 → -2,2 | 43,1 → 49,5 | 78,7 → 89,6 | 55,5 → 53,0 |
| H2 | 201 | 14,5 → 14,4 | 9,7 → 9,7 | -5,3 → -2,2 | 52,7 → 48,3 | 82,1 → 88,1 | 58,7 → 65,0 |
| H3 | 200 | 16,6 → 15,6 | 11,3 → 11,0 | -5,7 → -1,7 | 49,5 → 52,5 | 81,0 → 88,5 | 61,0 → 90,9 |
| H4 | 199 | 17,2 → 17,8 | 12,0 → 12,2 | -2,2 → -1,9 | 55,3 → 48,7 | 79,4 → 89,4 | 70,9 → 110,3 |
| H5 | 198 | 23,7 → 20,0 | 17,1 → 13,3 | -7,2 → -3,1 | 44,4 → 50,0 | 75,3 → 89,4 | 67,3 → 121,6 |
| H6 | 197 | 23,5 → 20,0 | 16,0 → 13,4 | -8,3 → -5,5 | 42,1 → 51,8 | 71,1 → 89,3 | 61,4 → 124,7 |
| H7 | 196 | 24,9 → 21,6 | 17,3 → 14,1 | -12,9 → -7,0 | 39,3 → 51,0 | 65,8 → 89,8 | 53,9 → 128,8 |
| H8 | 195 | 25,8 → 22,6 | 18,7 → 15,1 | -16,3 → -8,7 | 35,9 → 48,7 | 60,0 → 89,2 | 49,3 → 135,7 |

As metas nominais ficaram dentro dos intervalos descritivos de 95% obtidos por reamostragem em blocos de oito semanas para as 32 combinações de município, horizonte e nível de cobertura. A cobertura observada por temporada variou substancialmente. [Coberturas e incerteza por horizonte](../logs/growth-calibrado/coverage_uncertainty.csv)

## Pico do primeiro fold

As duas cidades atingiram o máximo observado na SE 202315. A tabela compara previsões emitidas para essa semana em diferentes antecedências. O MAE da última coluna usa todas as semanas do décimo superior do fold de 2022–23, incluindo empates.

| Cidade | Horizonte | Observado | Mediana anterior | Mediana candidata | Faixa 90% candidata | MAE nas semanas altas |
| --- | --- | --- | --- | --- | --- | --- |
| Campo Grande | H1 | 1132 | 816 | 1035 | 731–1397 | 228,2 → 113,7 |
| Campo Grande | H2 | 1132 | 795 | 1170 | 644–1773 | 243,5 → 106,5 |
| Campo Grande | H3 | 1132 | 481 | 864 | 347–1639 | 483,2 → 114,0 |
| Campo Grande | H4 | 1132 | 508 | 927 | 227–2208 | 466,0 → 135,2 |
| Campo Grande | H5 | 1132 | 459 | 773 | 94–2231 | 588,0 → 222,3 |
| Campo Grande | H6 | 1132 | 345 | 644 | 40–2206 | 633,3 → 274,5 |
| Campo Grande | H7 | 1132 | 349 | 599 | 21–2352 | 618,7 → 322,5 |
| Campo Grande | H8 | 1132 | 618 | 775 | 0–3695 | 428,3 → 290,0 |
| Dourados | H1 | 336 | 195 | 221 | 125–377 | 79,4 → 67,7 |
| Dourados | H2 | 336 | 225 | 239 | 79–509 | 55,9 → 58,7 |
| Dourados | H3 | 336 | 174 | 285 | 9–837 | 70,4 → 38,4 |
| Dourados | H4 | 336 | 204 | 229 | 0–936 | 60,3 → 62,4 |
| Dourados | H5 | 336 | 141 | 211 | 0–1147 | 138,6 → 72,1 |
| Dourados | H6 | 336 | 113 | 205 | 0–1222 | 126,1 → 78,3 |
| Dourados | H7 | 336 | 87 | 161 | 0–1004 | 158,0 → 101,7 |
| Dourados | H8 | 336 | 128 | 161 | 0–1308 | 158,1 → 123,0 |

O pico do primeiro fold entrou na faixa de 90% de H1 a H8 nas duas cidades. Em H8 a mediana ainda subestima o pico, e a faixa pode ser ampla o suficiente para limitar sua utilidade. As curvas por semana permitem inspecionar tanto a amplitude quanto o momento da subida. [Campo Grande](../logs/growth-calibrado/campo_grande_folds.png) · [Dourados](../logs/growth-calibrado/dourados_folds.png)

No fold parcial de 2025–26 em Campo Grande, a faixa de 50% cobriu 30,4% e a de 90% cobriu 93,7%. As médias dos quatro folds escondem essa variação.

No fold parcial de 2025–26 em Dourados, a faixa de 50% cobriu 47,0% e a de 90% cobriu 83,4%. As médias dos quatro folds escondem essa variação.


## Quanto o ganho varia entre temporadas

Valores positivos indicam redução do erro. Os intervalos descritivos de 95% usam 3.000 reamostragens em blocos circulares de oito semanas de origem dentro de cada temporada. Todos os horizontes da mesma origem entram juntos. As quatro temporadas também orientaram a escolha entre variantes, o que limita a inferência sobre temporadas futuras.

| Cidade | Horizontes | Métrica | Redução média | Intervalo de 95% | Folds com melhora |
| --- | --- | --- | --- | --- | --- |
| Campo Grande | H1–H8 | MAE | 19,8 | 5,4 a 37,5 | 3/4 |
| Campo Grande | H1–H8 | WIS | 14,3 | 5,9 a 24,2 | 4/4 |
| Campo Grande | H5–H8 | MAE | 23,6 | 5,1 a 45,6 | 3/4 |
| Campo Grande | H5–H8 | WIS | 20,3 | 7,3 a 34,7 | 4/4 |
| Dourados | H1–H8 | MAE | 1,9 | -0,9 a 4,7 | 3/4 |
| Dourados | H1–H8 | WIS | 1,8 | 0,0 a 3,5 | 4/4 |
| Dourados | H5–H8 | MAE | 3,5 | -0,7 a 7,8 | 3/4 |
| Dourados | H5–H8 | WIS | 3,3 | 0,4 a 6,5 | 4/4 |

## Treino e calibração

O candidato combina o LightGBM municipal com dois modelos diretos por horizonte. A regressão de crescimento utiliza mudanças nos casos em janelas de duas a 16 semanas, a época do ano e a tendência da outra cidade com atraso de uma semana. O modelo regional ajusta cinco quantis com séries de dez municípios. Quando as trajetórias municipal e regional sobem juntas, a combinação inicial dá mais peso à tendência. O calibrador também compara o erro histórico dos componentes e de uma extrapolação simples de crescimento amortecido.

Os treinos municipais completos usam cinco sementes, cinco quantis e oito horizontes. Os folds de setembro de 2019 a agosto de 2022 geram previsões fora da amostra para iniciar a calibração. Em cada fold avaliado, um erro só entra na calibração quando a semana-alvo terminou e transcorreram as 16 semanas de maturação. Os modelos de base ficam congelados no início do fold. A cada origem, o calibrador atualiza a mediana e os intervalos com os erros já maduros. Ele dá mais peso a origens com horizonte, nível de casos, tendência, variabilidade e época do ano semelhantes. Para estimar o tamanho efetivo da amostra, agrupa os pesos das previsões que compartilham a mesma semana-alvo.

As oito séries auxiliares de dengue do InfoDengue são de Aquidauana, Corumbá, Maracaju, Naviraí, Nova Andradina, Ponta Porã, Sidrolândia e Três Lagoas. O experimento congelou as séries junto às duas bases municipais, configurações e históricos de referência. Revisões posteriores à emissão podem alterar as contagens das origens históricas. A coorte prospectiva segue separada.

## Reprodução e evidências

Na raiz do repositório, execute o comando abaixo com um diretório de saída novo. Ele congela as entradas, treina os modelos e grava previsões por semana, horizonte e fold. O diretório desta avaliação está em [logs/growth-calibrado](../logs/growth-calibrado/).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.train_growth_forecasts --output logs/growth-replica
```

Para reavaliar os componentes com os dados e os quatro treinos municipais congelados desta execução, use o comando abaixo.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.train_growth_forecasts --evaluate-existing --output logs/growth-calibrado
```

Os CSVs de previsões incluem uma linha por método e identificam a origem, o alvo, a maturidade, os cinco quantis e a contagem observada. As métricas incluem a referência histórica, o retreino exclusivamente municipal e a combinação final. [Métricas por fold e horizonte](../logs/growth-calibrado/metrics.csv) · [Variantes testadas](../logs/growth-calibrado/experiments.csv) · [Picos por temporada](../logs/growth-calibrado/peaks.csv) · [Integridade dos arquivos](../logs/growth-calibrado/manifest.json)
