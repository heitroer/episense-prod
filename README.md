# EpiSense

**Previsão semanal de casos de dengue por município.**

O EpiSense estima o volume de casos de dengue para as próximas oito semanas epidemiológicas. As previsões são disponibilizadas em um painel e por API, com faixas de incerteza para apoiar a interpretação dos resultados.

## Cobertura municipal

O ambiente está configurado para três municípios de Mato Grosso do Sul. A disponibilidade das previsões depende dos dados e dos artefatos operacionais de cada município.

| Município | Código IBGE |
| --- | --- |
| Campo Grande | `5002704` |
| Dourados | `5003702` |
| Corumbá | `5003207` |

## O que o EpiSense entrega

- **Painel municipal** com histórico de casos e previsões para consulta por semana epidemiológica.
- **API autenticada** para integrar as previsões e seus intervalos a outras ferramentas.
- **Histórico de emissões** para consultar as previsões registradas e acompanhar seu desempenho à medida que os dados de referência se consolidam.

## Interpretação das previsões

A estimativa central indica o volume esperado de casos. As faixas expressam a incerteza associada a cada horizonte. A leitura deve considerar a semana de emissão, a semana prevista e a possibilidade de revisão dos dados epidemiológicos.

O desempenho varia por município e horizonte. A avaliação retrospectiva descreve o desempenho em períodos históricos. A avaliação prospectiva acompanha as previsões registradas antes de os resultados estarem disponíveis.

## Execução local

Requer Docker com Compose e os dados e artefatos de previsão provisionados para os municípios que serão consultados.

1. Copie o arquivo de exemplo apenas se ainda não houver um `.env` local.

   ```bash
   cp -n .env.example .env
   ```

2. Configure um token forte em `EPISENSE_API_KEY` no `.env` e inicie os serviços.

   ```bash
   docker compose up -d --build
   ```

3. Acesse o painel em <http://localhost:8501> e informe o token configurado. A API atende em <http://localhost:8001>. Use `GET /health` para consultar o estado da API.

### Consulta à API

Com `EPISENSE_API_KEY` disponível no ambiente do terminal, envie o código IBGE no campo `geocode`.

```bash
curl --fail-with-body http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${EPISENSE_API_KEY}" \
  --data '{"geocode":"5002704"}'
```

Mantenha o `.env` fora do controle de versão e use HTTPS em ambientes expostos à rede.
