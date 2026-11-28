# Publicação dos modelos de dengue em Campo Grande

O treino copia a base e a configuração antes de iniciar os workers. A versão guarda código-fonte do treino, versões de Python e pacotes, datas, hash da base e hashes dos artefatos. O motor carrega a configuração congelada das versões novas. Um arquivo de versão contém os 200 modelos, a previsão histórica por fold, os quantis pós-processados, a calibração e os relatórios. O diretório ativo é trocado atomicamente após os testes.

## Critérios em vigor

A política de engenharia compara o candidato à versão ativa em semanas maduras comuns. Exige pelo menos 104 semanas e três picos em cada horizonte. Tolera até 2% de aumento no MAE e WIS e até 10% no erro de pico e na largura média. Exige cobertura de pelo menos 70% e permite no máximo dois pontos percentuais adicionais de distância ao nominal de 90%. O orçamento de MAE frente à persistência é de 1,10 vez em H1/H2, 1,30 em H3 e 1,00 em H4–H8. Os números são limites provisórios de regressão de engenharia. Uma versão aprovada pode continuar com cobertura inferior à nominal.

O histórico pós-processado é pontuado novamente a partir das contagens da base para conferir paridade entre dados servidos e métricas. O teste isolado de concorrência, falha de treino, reprovação por WIS e divergência introduzida na cobertura passou em Docker. O comparador encontrou de 195 a 202 pares por horizonte e cinco semanas de pico em cada um. A cobertura inclusiva usa a contagem inteira observada, inclusive quando coincide exatamente com um limite.

Um treino completo anterior foi publicado somente no diretório de teste `logs/gate-run-3sqfwx7l`. Para reproduzir o teste de falhas em diretório temporário:

```sh
docker compose exec -T episense-dashboard python scripts/check_release_policy.py /app/models/releases/model-f7suo09u
```

Em 22/11/2026, o treino completo de produção foi aprovado e ativou a versão `model-aj6h73qj`, referente à SE 202645. A verificação pós-publicação conferiu todos os hashes do manifesto e contou 200 modelos e 1588 previsões maduras avaliadas.
