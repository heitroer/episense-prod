# Diagnóstico dos intervalos de oito semanas em Campo Grande

A release `model-f7suo09u` cobriu 147 de 195 desfechos maduros (75,4%) com o intervalo nominal de 90%. Foram 32 observações acima do q95 e 16 abaixo do q05. As falhas mudam de lado entre temporadas. A release anterior registrou 74,9% porque tratou 16,999999999999996 como inferior ao limite de 17 casos na SE 202629. A contagem inteira da base é 17 e está dentro da faixa inclusiva. A largura e o WIS reproduzem o relatório da release.

| Grupo | Semanas | Abaixo do q05 | Acima do q95 | Cobertura |
|---|---:|---:|---:|---:|
| Fold 0 | 52 | 4 | 25 | 44,2% |
| Fold 1 | 52 | 4 | 7 | 78,8% |
| Fold 2 | 53 | 0 | 0 | 100,0% |
| Fold 3 | 38 | 8 | 0 | 78,9% |
| Alta incidência no alvo | 48 | 0 | 14 | 70,8% |
| Baixa incidência no alvo | 147 | 16 | 18 | 76,9% |

A alta incidência usa o percentil 75 do histórico disponível no primeiro corte (22,40 casos por 100 mil). O grupo é definido pelo desfecho observado e serve exclusivamente ao diagnóstico retrospectivo. Nos alvos de alta, cada falha foi uma subestimação do limite superior. Entre alvos de baixa ocorreram erros nos dois sentidos. O primeiro fold concentrou 25 dos 32 excessos acima do q95.

## Teste temporal de uma correção assimétrica

Uma única candidata ajustou as duas caudas separadamente. Para cada cauda, calculamos um deslocamento em log(1 + casos) a partir do erro unilateral entre o limite previsto e o observado. Usamos o valor de ordem 128 entre 133 erros anteriores já maduros na origem 202537, equivalente ao percentil unilateral de 95% com arredondamento para cima de 0,95 × (n + 1). Limitamos cada deslocamento ao sentido de expansão. Aplicamos os deslocamentos congelados (0,583 inferior e 0,712 superior) às 38 origens do último fold, da SE 202537 à 202621, preservando os três quantis centrais. A disponibilidade dos desfechos segue o embargo de 16 semanas do projeto.

| Medida nas mesmas 38 semanas | Publicado | Candidato |
|---|---:|---:|
| Resultados dentro do intervalo | 30 | 36 |
| Abaixo do limite inferior | 8 | 2 |
| Acima do limite superior | 0 | 0 |
| Largura média, em casos | 198,74 | 456,05 |
| WIS médio | 43,98 | 47,64 |
| MAE da mediana | 70,79 | 70,79 |

**Decisão.** A candidata trouxe a cobertura para 94,7%, aumentou a largura em 129,5% e piorou o WIS em 8,3%. A cauda superior foi alargada por erros das temporadas antigas e nenhum desfecho excedeu o teto publicado na temporada de teste. A candidata foi rejeitada para produção. A cobertura de H8 continua pendente de solução operacional.

O último fold ficou fora do cálculo dos deslocamentos. Seus resultados agregados já haviam sido examinados em experimentos anteriores. Os dados são consolidados e a incidência muda entre anos. Para a próxima decisão, congelar qualquer nova regra antes das emissões e registrar, em paralelo, limite por limite e por origem, para avaliação em semanas novas. O desfecho da semana-alvo pode ser usado para pontuar depois da maturação, mas é indisponível para selecionar a calibração no momento da emissão.

## Reprodução

```sh
.venv/bin/python scripts/diagnose_h8_tails.py models/releases/model-f7suo09u
```

O script usa o histórico pós-processado, o snapshot de treino e a avaliação da própria release. Ele valida o embargo registrado, a unicidade das semanas-alvo, a ordenação dos quantis e a igualdade dos indicadores agregados com a avaliação publicada.

## Sensibilidade à temporada recente

Após a análise acima, testamos uma segunda hipótese exploratória nas mesmas 38 origens. Em cada emissão simulada, usamos até 52 erros relativos à mediana já maduros naquela origem, com peso n/(n+26) nos quantis de ordem 5% e 95% em log(1+casos). O restante do peso ficou com a faixa publicada. Essa candidata estreitou a faixa média para 139,32 casos, deixou 19 observações abaixo do q05 e cobriu somente 50%. Seu WIS foi 44,99, ante 43,98 da versão publicada. A regra recente também foi rejeitada. O atraso de disponibilidade limita a rapidez com que os erros recentes podem orientar a calibração. Nenhum dos dois testes constitui evidência prospectiva.
