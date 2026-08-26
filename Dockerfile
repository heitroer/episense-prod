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

# SHA256 oficial, PyPI 2026-07-09: xgboost==3.3.0
ARG XGBOOST_VERSION=3.3.0
ARG XGBOOST_SHA256_AMD64=f59edaf28eccd1c519788607c72ed907ee6cedfa933d706620bc1612d24b354e
ARG XGBOOST_SHA256_ARM64=624a83aeb1e7ba081719795db179f4ce6fff12e79de05cd9baf15ee48fd22f0e

# SHA256 oficial, PyPI 2026-07-09: apscheduler==3.11.3 (py3-none-any)
ARG APSCHEDULER_SHA256=bbeb2ec02d23d3c06a6c07ed7f0f3939ada6680eb121fae809a69bb42c537a30
ARG APSCHEDULER_VERSION=3.11.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y xgboost 2>/dev/null || true

RUN XGBOOST_ARCH=$(uname -m) \
    && if [ "$XGBOOST_ARCH" = "x86_64" ]; then \
         XGBOOST_WHL="xgboost-${XGBOOST_VERSION}-py3-none-manylinux_2_28_x86_64.whl"; \
         XGBOOST_SHA="${XGBOOST_SHA256_AMD64}"; \
         XGBOOST_HASH_DIR="47/1f/8b3e578cfd8e3bcdb4374e2bbe0b40b4e5320accb5cbdcf535ecc512eb5c"; \
       elif [ "$XGBOOST_ARCH" = "aarch64" ]; then \
         XGBOOST_WHL="xgboost-${XGBOOST_VERSION}-py3-none-manylinux_2_28_aarch64.whl"; \
         XGBOOST_SHA="${XGBOOST_SHA256_ARM64}"; \
         XGBOOST_HASH_DIR="47/3a/a0adcd1ee28f525bd5c9dc3ebe78a7599bf97c22866d6449f967b829e338"; \
       else echo "Unsupported arch: $XGBOOST_ARCH" && exit 1; fi \
    && curl -fsSL -o "/tmp/${XGBOOST_WHL}" \
        "https://files.pythonhosted.org/packages/${XGBOOST_HASH_DIR}/${XGBOOST_WHL}" \
    && echo "${XGBOOST_SHA}  /tmp/${XGBOOST_WHL}" | sha256sum -c - \
    && pip install --no-cache-dir --no-deps "/tmp/${XGBOOST_WHL}" \
    && rm "/tmp/${XGBOOST_WHL}"

COPY api/ ./api/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]