"""Disponibilidade de contagens observadas na grade epidemiológica."""


def observed_values(frame, column="casos"):
    """Preserva a grade semanal e remove preenchimentos apenas da série de alvos."""
    values = frame[column]
    if "is_imputed" in frame:
        values = values.mask(frame["is_imputed"].fillna(False).astype("boolean"))
    if "casos_observados" in frame:
        values = values.where(frame["casos_observados"].notna())
    return values
