#!/bin/bash
# Episense auto-update (cron): coleta InfoDengue + OpenMeteo, reprocessa a base
# e retreina os modelos se houver dados novos. Saida curta para o cron.
cd "$(dirname "$0")/.." || exit 1
exec /usr/bin/env python3 scripts/update_pipeline.py
