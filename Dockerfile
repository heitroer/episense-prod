FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# SUPPLY-CHAIN: verificacao SHA256 manual das wheels sem --require-hashes
#
# Alguns pacotes (xgboost, apscheduler) tem deps CUDA transitivas que o
# --require-hashes do pip nao consegue verificar. Para nao perder protecao
# de supply-chain, fazemos o download manual da wheel do PyPI (via HTTPS)
# e validamos o SHA256 oficial ANTES de instalar.
#
# Como atualizar os hashes ao subir versao:
#   curl -s https://pypi.org/pypi/xgboost/$VERSAO/json \
#     | jq -r '.urls[] | select(.filename | contains("manylinux_2_28_x86_64"))
#              | "\\(.digests.sha256)  \\(.url)"'
#   curl -s https://pypi.org/pypi/apscheduler/$VERSAO/json \
#     | jq -r '.urls[] | select(.filename | contains("py3-none-any"))
#              | "\\(.digests.sha256)  \\(.url)"'
# Cole o hash e URL abaixo.
# ==============================================================================

# SHA256 oficial, PyPI 2026-07-09: apscheduler==3.11.3 (py3-none-any)
# Nota supply-chain (2026-09-07): xgboost==3.3.0 removido de requirements.txt após grep confirmar 0 imports
# (xgboost, optuna*, slowapi, limits, cachetools, httpx, email-validator não importados); bloco SHA/uninstall xgboost removido.
ARG APSCHEDULER_SHA256=bbeb2ec02d23d3c06a6c07ed7f0f3939ada6680eb121fae809a69bb42c537a30
ARG APSCHEDULER_VERSION=3.11.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]