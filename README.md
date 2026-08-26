# Episense Production

API de previsão de casos de dengue para Campo Grande, MS usando dados em tempo real das APIs InfoDengue e OpenMeteo.

## Características

- **Dados em tempo real**: Coleta dados frescos das APIs InfoDengue e OpenMeteo a cada requisição (cache de 5 min)
- **Modelo LightGBM ensemble**: 5 seeds × 8 horizontes (h1-h8)
- **Walk-forward validation**: 5 folds (2021-2025), gap 8 semanas
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
├── scripts/
│   ├── train.py             # Script de treinamento principal
│   └── update_pipeline.py   # Pipeline de atualização (cron)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Build e subir
docker compose up -d --build

# Health check
curl http://localhost:8001/health

# Previsão
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"geocode": "5002704"}'

# Métricas
curl http://localhost:8001/metrics

# Metadados
curl http://localhost:8001/metadata
```

## Treinamento (offline)

```bash
# Requer dados históricos completos (CSVs)
python scripts/train.py
```

## Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Status da API e modelos |
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

## Métricas de Validação (Walk-Forward 2021-2025)

| Horizonte | RMSE | WMAPE | F2-Score | PR-AUC | R² |
|-----------|------|-------|----------|--------|-----|
| h1 | ~43-170 | ~0.19-0.32 | 0.78-0.97 | 0.90-0.99 | 0.45-0.85 |
| h4 | ~26-85 | ~0.19-0.24 | 0.93-0.94 | 0.97-0.98 | 0.80-0.89 |
| h8 | ~150-175 | ~0.45-0.47 | 0.87-0.94 | 0.91 | 0.45-0.65 |

## Cron Job (Atualização Diária)

O script `scripts/update_pipeline.py` pode ser agendado via cron para:
1. Coletar dados novos do InfoDengue + OpenMeteo
2. Reprocessar base
3. Retreinar apenas se a base mudou
4. Restart da API

```bash
# Exemplo crontab (06:00 diário)
0 6 * * * cd /path/to/Episense-Prod && python scripts/update_pipeline.py
```