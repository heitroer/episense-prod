"""
Calendário epidemiológico brasileiro — semana começa no DOMINGO.

Regra (validada contra os dados do InfoDengue 2011–2026, 811 semanas):
- A semana 1 do ano Y é a semana (domingo a sábado) que contém 4 de janeiro
  (equivalente: a semana que contém 1º de janeiro se ≥4 dias caírem no ano novo).
- O ano Y tem 53 semanas quando w1(Y+1) − w1(Y) == 371 dias.
  Anos com 53 semanas no período: 2014, 2020, 2025 (NÃO os anos ISO 2015/2026).
- SE no formato YYYYWW (string de 6 dígitos). NUNCA usar isocalendar() (ISO
  começa na segunda-feira e numera as semanas de forma diferente).
"""

import pandas as pd


def _w1_sunday(year: int) -> pd.Timestamp:
    """Domingo de início da semana epidemiológica 1 do ano (contém 4 de janeiro)."""
    jan4 = pd.Timestamp(year, 1, 4)
    return pd.Timestamp(jan4 - pd.Timedelta(days=(jan4.dayofweek + 1) % 7))


def weeks_in_year(year: int) -> int:
    """Número de semanas epidemiológicas do ano (52 ou 53, calendário brasileiro)."""
    return int((_w1_sunday(year + 1) - _w1_sunday(year)).days // 7)


def epiweek_to_date(year: int, week: int) -> pd.Timestamp:
    """Domingo de início da semana epidemiológica (year, week)."""
    return pd.Timestamp(_w1_sunday(year) + pd.Timedelta(weeks=week - 1))


def date_to_epiweek(d) -> str:
    """Converte uma data para SE (YYYYWW) no calendário epidemiológico brasileiro.

    Datas fora da faixa do ano civil pertencem à semana 1 do ano seguinte
    (ex.: 2024-12-29, domingo de início da 202501) ou à última semana do ano
    anterior (ex.: 2011-01-01 -> 201052).
    """
    d = pd.Timestamp(d)
    y = int(d.year)
    w1 = _w1_sunday(y)
    if d < w1:
        y -= 1
        w1 = _w1_sunday(y)
    elif d >= _w1_sunday(y + 1):
        y += 1
        w1 = _w1_sunday(y)
    week = (d - w1).days // 7 + 1
    return f"{y}{week:02d}"
