# Acompanhamento prospectivo de dengue em Campo Grande

## Resultado de 21/11/2026

O arquivo contém uma emissão oficial desde a entrada em operação na origem SE 202645. Até o corte havia uma origem esperada, nenhuma semana perdida e nenhum par H1/H2 incompleto. A execução da rotina de reprodução dentro do Docker recuperou os quantis da emissão pelo modelo e pelas entradas arquivadas. Há 27 registros legados sem hash de modelo. Uma emissão verificada permite testar a trilha de auditoria. A continuidade do agendamento exige várias semanas de observação.

| Horizonte | Semana-alvo | Mediana prevista | Persistência | Antecedência em relação ao início do alvo | Primeira coleta madura elegível |
|---|---|---:|---:|---|---|
| H1 | 202646 | 53 | 1 | Emitida 6 dias após o início | 14/03/2027 |
| H2 | 202647 | 58 | 1 | Emitida 1 dia antes do início | 21/03/2027 |

As duas comparações aguardam coleta madura. As datas seguem a regra operacional de 16 semanas após o encerramento do alvo. Os 23 registros semanais distintos recuperados de coletas datadas até o corte ainda não fornecem desfecho maduro para esses alvos. Para avaliar desempenho, o relatório calcula o MAE do modelo e da persistência nas mesmas semanas, separando as emissões anteriores à semana-alvo das emitidas durante a semana.

## Reprodução no Docker

Na raiz do projeto, o comando abaixo reproduz o corte. Para avaliar o arquivo na data corrente, omitir o argumento de data.

```sh
docker compose exec -T episense-dashboard python scripts/evaluate_published_forecasts.py --as-of 2026-11-21 --replay
```

No JSON, `operational_monitor` enumera origens esperadas desde a entrada em operação. Para a origem ainda em janela de emissão, a ausência aparece em `open_origins`. Após a virada semanal, uma origem ausente aparece em `missed_origins`. Cada alvo traz antecedência, data de maturação, última coleta recuperada e estado. Após a maturação, `missing_mature_collection` identifica que a série arquivada ainda precisa de uma coleta daquele alvo em data elegível. Registros oficiais duplicados geram erro para evitar inflar a amostra.

O intervalo consultado rotineiramente no InfoDengue cobre 126 dias e a coleta depende de o serviço estar em execução. Perder todas as coletas elegíveis enquanto uma semana ainda está nessa janela pode deixar uma lacuna. A versão armazenada em cada arquivo datado permite avaliar o estado conhecido ao final daquela data.

## Corte de 22/11/2026

Havia duas origens esperadas desde o início do monitoramento. A SE 202645 estava emitida e reproduzível. A SE 202646 estava em janela aberta. Às 21h30 de Campo Grande, a tentativa de emissão encontrou dados completos apenas até 202645 e registrou a pendência. Nenhuma semana perdeu a janela até esse corte. Continuam zero pares maduros. A partir das próximas emissões, a avaliação também registra e pontua separadamente a variante H8 sem o limitador superior, se houver previsão anterior ao alvo e coleta madura. Em uma simulação com as entradas da SE 202645, o q95 oficial de 183 casos passou a 288 nessa variante. O teste preservou a emissão original e não foi incluído na amostra prospectiva.
