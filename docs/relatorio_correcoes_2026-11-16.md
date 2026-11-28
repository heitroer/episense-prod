# Correções prioritárias do Episense

O [diagnóstico técnico e estratégico](diagnostico_estrategico_tecnico_2026-11-16.md) contém o inventário e a análise de fundo. Este relatório acompanha as correções no checkout.

| Problema | Impacto | Correção prevista |
|---|---|---|
| Reprojeção automática do arquivo de previsões | O gráfico atribui a origens antigas resultados gerados com o modelo atual | Preservar os registros originais e parar a reprojeção e o preenchimento retroativo |
| Preenchimento de lacunas por inferência retroativa | Uma semana sem evidência ganha uma previsão aparente | Manter lacunas e identificar a origem dos pontos exibidos |
| Calibração com o conjunto completo no gráfico de backtest | O gráfico usa parâmetros ajustados com os próprios períodos exibidos | Mostrar o backtest com os quantis brutos registrados |
| Promoção aceita falha na recalibração ou métricas | Artefatos podem ficar inconsistentes | Rejeitar e restaurar o conjunto anterior |
| Gate ausente ou inválido aceita retreino | Falhas na validação podem passar despercebidas | Exigir um resultado estruturalmente válido |

As correções da tabela foram implementadas. O dashboard identifica a origem do ponto, mantém os quantis originais disponíveis no arquivo legado e registra a primeira previsão completa da origem atual devolvida pelo serviço, com horário em UTC. Registros antigos têm indicação de arquivo legado porque seu momento de emissão não foi verificado. O rollback restaura os arquivos do backup e remove artefatos criados pelo candidato. O cron cancela o treino quando a cópia de segurança falha.

A verificação local usou os registros reais em leitura e arquivos temporários para testar imutabilidade, lacunas históricas, rejeição de gate incompleto e restauração após falha simulada no recálculo. O teste do rollback não executou um retreino completo.

## Pendências

- **Publicação atômica de modelos.** Os artefatos ainda são escritos no diretório monitorado durante o treino. O rollback protege o estado ao final de uma falha. A troca segura durante a execução exige treinar em uma versão isolada e publicar um apontador único.
- **Paridade de validação com inferência.** O histórico bruto evita aplicar parâmetros futuros ao gráfico. As métricas da cadeia completa ainda exigem uma função compartilhada e avaliação temporal com os parâmetros e dados disponíveis na origem.
- **Dados e modelo.** Faltam snapshots com procedência por emissão, manifesto de versão do modelo, avaliação contra persistência e verificação do atraso de notificação.
- **Operação institucional e expansão.** Permanecem a revisão do controle de acesso e da infraestrutura, a parametrização por município e doença e a validação das novas séries.

## Atualização da publicação de modelos

A pendência de publicação atômica foi corrigida no cron, no dashboard e no comando padrão de treino. Cada execução usa uma cópia fixa da base. O candidato só pode ser ativado após passar pela validação, pelo recálculo e pela conferência dos modelos esperados. API e dashboard resolvem o apontador ativo e leem os artefatos dessa versão. A instalação atual continua usando sua versão existente até a primeira publicação pelo novo fluxo.

O teste em diretório temporário reutilizou os 200 modelos existentes e executou a recalibração e o recálculo de métricas. Também simulou uma falha e verificou que a versão ativa foi preservada. A execução completa de um novo treino permanece por verificar.

A cópia fixa resolve a consistência entre etapas de um mesmo retreino. Ainda faltam reconstruir os dados disponíveis na data de cada previsão histórica, avaliar a paridade da cadeia de inferência e comparar com persistência. A expansão estadual e as demais doenças permanecem frentes separadas.

## Avaliação das previsões publicadas

O dashboard passou a registrar a entrada da inferência, o hash dos artefatos, a configuração do modelo e a última contagem observada disponível antes do ajuste por atraso de notificação. A reprodução chama a mesma função de inferência e a mesma conversão de quantis usadas na publicação. A comparação com persistência usa exatamente as mesmas semanas-alvo por horizonte e informa o número de pares, MAE do modelo, MAE da persistência, ganho relativo e métricas dos intervalos. O protocolo exclui alvos cuja semana já havia começado na data da emissão e aguarda, por padrão, 16 semanas após o encerramento do alvo.

O comando também reconstrói o que está disponível nos arquivos datados de casos e clima até a data especificada. A reconstrução usa a última contagem disponível por semana, mantém as datas das coletas e só inclui semanas de clima completas. A opção de exportação permite inspecionar a procedência de cada linha.

```sh
.venv/bin/python scripts/evaluate_published_forecasts.py --replay
.venv/bin/python scripts/evaluate_published_forecasts.py --as-of 2026-11-16 --vintage-csv /tmp/episense-vintage.csv
```

Em 16/11/2026 havia 22 semanas recuperáveis com casos e clima completo. Os 27 registros antigos carecem da entrada exata e do hash do modelo. O avaliador não atribui a eles uma reprodução nem calcula ganho prospectivo com essa amostra. O histórico de 2022 a 2025 continua sendo um backtest com dados consolidados. As coletas disponíveis não permitem refazê-lo integralmente com as informações originais. A nova avaliação acompanha as emissões arquivadas pelo dashboard e acumula evidência à medida que os alvos amadurecem.
