"""Resolve a versão de modelos ativa, com fallback para instalações existentes."""
from pathlib import Path


def active_models(root: Path) -> Path:
    return (root / "current").resolve(strict=True) if (root / "current").is_symlink() else root
