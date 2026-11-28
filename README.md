# Episense Production

API de previsão de casos de dengue para Campo Grande, Dourados e Corumbá, MS usando dados em tempo real das APIs InfoDengue e OpenMeteo.

A série de referência do treino e das previsões usa o nowcast semanal do InfoDengue (`casos_est`) em todo o histórico municipal. Quando a estimativa falta ou é inválida, o processamento usa a contagem notificada da própria semana (`casos` na fonte). A base processada guarda a referência em `casos` e a notificação original em `casos_notificados`. O dashboard mostra as duas séries e informa a contagem confirmada (`casconf`) quando disponível. As avaliações com dados históricos revisados podem incorporar revisões posteriores à data de cada previsão.

## Características

- **Dados em tempo real**: Coleta dados frescos das APIs InfoDengue e OpenMeteo a cada requisição (cache de 5 min)
- **Modelo LightGBM ensemble**: 5 seeds × 8 horizontes (h1-h8)
- **Walk-forward validation**: 4 folds (2022-2025), gap 8 semanas
- **Calendário epidemiológico brasileiro**: Semana começa domingo, semana 1 contém 4/jan, anos de 53 semanas (2014, 2020, 2025)
- **Features**: 300 por horizonte (lags, rolling, sazonalidade YoY 52/53, clima longo, interações)

## Estrutura

```
Episense-Prod/
├── api/
│   └── main.py              # API FastAPI (coleta dados nas APIs na requisição)
├── config/
│   ├── config.yaml          # Configuração principal
│   └── config.py            # Loader de config
├── data/
│   ├── collect_infodengue.py
│   ├── collect_openmeteo.py
│   ├── epiweeks.py          # Calendário epi brasileiro
│   ├── features.py          # Feature engineering
│   ├── features_exp.py      # Features experimentais
│   └── process_data.py      # Pipeline de processamento
├── models/
│   ├── inference.py         # Motor de inferência
│   ├── lgbm_h*_seed*.pkl    # 40 modelos (8 horizontes × 5 seeds)
│   ├── feature_list.json    # 300 features/horizonte
│   ├── normalization_params.json
│   └── validation_results.json
├── dashboard/
│   └── app.py               # Dashboard FastAPI + frontend estático
├── scripts/
│   ├── train.py             # Script de treinamento principal
│   └── update_pipeline.py   # Pipeline de atualização (cron)
├── episense_security.py     # Token de API/dashboard, CORS e rate limit compartilhados
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                     # Segredos locais (não versionado)
├── .env.example             # Modelo de configuração
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Configure os segredos (o .env não é versionado)
cp .env.example .env

# Gere um valor forte e cole no .env:
python -c "import secrets; print(secrets.token_hex(32))"  # EPISENSE_API_KEY

# Carregue no shell para os exemplos abaixo
set -a; . ./.env; set +a

# 2. Build e subir
docker compose up -d --build

# Health check (público)
curl http://localhost:8001/health

# Previsão (autenticada por X-API-Key ou Authorization: Bearer)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EPISENSE_API_KEY" \
  -d '{"geocode": "5002704"}'

# Métricas municipais (também disponível em /metrics/walk-forward e /metadata)
curl -H "X-API-Key: $EPISENSE_API_KEY" "http://localhost:8001/metrics?geocode=5003207"

# Metadados
curl -H "X-API-Key: $EPISENSE_API_KEY" http://localhost:8001/metadata
```

O campo `geocode` aceita `5002704` (Campo Grande), `5003702` (Dourados) e `5003207` (Corumbá). Os serviços municipais leem a configuração, a base, os modelos e o histórico de emissões de seus próprios volumes. Para Corumbá, a raiz local desses volumes é `cities/5003207/`. O limiar epidêmico vem do [relatório InfoDengue de Corumbá da SE 202646](https://info.dengue.mat.br/report/MS/5003207/202646).

O dashboard fica em `http://localhost:8501`. Ao abrir, ele pede apenas o
**token da API** (`EPISENSE_API_KEY`) numa tela própria e o envia como
`X-API-Key`; não há usuário/senha nem popup do navegador. O token fica no
`sessionStorage` (somente na aba atual) e pode ser trocado pelo botão
“Trocar token”.

Em `ENVIRONMENT=production` ambos os serviços **não sobem** sem um
`EPISENSE_API_KEY` forte. Em `development` a autenticação é opcional, mas
continua ativa se a variável estiver definida. O Swagger (`/docs`) fica
disponível apenas em `development` e mostra o botão **Authorize** com os
esquemas `X-API-Key` e Bearer.

## Segurança e configuração

Todas as variáveis são lidas do ambiente (ou do `.env` local, via
`python-dotenv`):

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `ENVIRONMENT` | `production` | `production` exige token forte e desativa `/docs`. `development` é somente local. |
| `EPISENSE_API_KEY` | — | Token da API e do dashboard (≥32 chars em produção), via `X-API-Key` ou `Authorization: Bearer`. |
| `EPISENSE_CORS_ORIGINS` | vazio | Origens exatas separadas por vírgula. `*` é rejeitado; vazio desativa CORS. |
| `EPISENSE_RATE_LIMIT_PER_MINUTE` | `120` | Limite por IP e por processo (`0` desativa). |
| `TRUSTED_PROXY_CIDR` | vazio | Redes de proxy autorizadas a enviar `X-Forwarded-For`/`X-Real-IP`. |

Notas:

- O rate limit é em memória e por processo; com `--workers 2` o limite efetivo
  dobra. Para limite global, use um reverse proxy ou um store compartilhado.
- Com portas publicadas diretamente pelo Docker, todos os clientes podem
  aparecer com o IP do gateway; configure `TRUSTED_PROXY_CIDR` apenas se houver
  um proxy reverso real assinando `X-Forwarded-For`. Caso contrário, deixe vazio.
- `/health` é público nos dois serviços (necessário para o healthcheck do
  Compose); as rotas `/api/*` do dashboard exigem o token.
- Placeholders como `replace-with-...` são rejeitados em produção.
- O token trafega no header `X-API-Key`; em produção use HTTPS (proxy reverso)
  para não expô-lo na rede.
- Nunca comite o `.env`; use o `.env.example` como referência.


## Treinamento (offline)

```bash
# Requer dados históricos completos (CSVs)
python scripts/train.py
```

## Endpoints

Todos exigem `X-API-Key` (ou `Authorization: Bearer`) exceto `/health`.

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Status da API e modelos (público) |
| `/predict` | POST | Previsão h1-h8 (JSON: `{"geocode": "5002704"}`) |
| `/predict/{geocode}` | GET | Previsão via path param |
| `/metrics` | GET | Métricas de validação walk-forward |
| `/metadata` | GET | Metadados do modelo e dados |

## Response Example

```json
{
  "geocode": "5002704",
  "data_previsao": "2026-08-25T19:37:28.157201",
  "previsoes": [
    {"horizonte_semanas": 1, "semana_epidemiologica": "202634", "casos_previstos": 24, "alerta": "medio"},
    {"horizonte_semanas": 2, "semana_epidemiologica": "202635", "casos_previstos": 21, "alerta": "medio"},
    ...
  ]
}
```

## Alertas

| Nível | h1-h4 | h5-h8 |
|-------|-------|-------|
| baixo | < 10 | < 10 |
| medio | 10-50 | 10-50 |
| alto | 50-100 | 50-100 |
| critico | > 100 | > 100 |

> **Nota**: h5-h8 são sinais de alerta precoce - menos confiáveis em anos de surto atípico (ex: 2019 com 45k casos).

## Métricas de Validação (Walk-Forward 2022-2025)

| Horizonte | RMSE | WMAPE | F2-Score | PR-AUC | R² |
|-----------|------|-------|----------|--------|-----|
| h1 | ~43-170 | ~0.19-0.32 | 0.78-0.97 | 0.90-0.99 | 0.45-0.85 |
| h4 | ~26-85 | ~0.19-0.24 | 0.93-0.94 | 0.97-0.98 | 0.80-0.89 |
| h8 | ~150-175 | ~0.45-0.47 | 0.87-0.94 | 0.91 | 0.45-0.65 |

## Atualização automática

O dashboard de cada município consulta o InfoDengue às **01h15, 07h15, 13h15 e 19h15** (America/Campo_Grande). Ao detectar uma nova semana, registra 16 contagens semanais, coleta o clima, processa a base, treina e emite oito horizontes após a publicação. A API e as páginas consultam a base local entre as coletas. O Docker precisa estar em execução em um host acordado nos horários agendados.
