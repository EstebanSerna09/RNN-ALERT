"""Empaquetado ZIP de los resultados del pipeline."""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def crear_zip(directorio: Path) -> Optional[bytes]:
    """
    Empaqueta todos los CSV de `directorio` (recursivo) en un ZIP en memoria.

    Returns:
        Bytes del ZIP, o None si no hay archivos.
    """
    archivos = list(directorio.rglob("*.csv"))
    if not archivos:
        logger.warning("No hay archivos CSV para empaquetar.")
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in archivos:
            zf.write(f, f.relative_to(directorio))

    buf.seek(0)
    logger.info(f"ZIP generado con {len(archivos)} archivos.")
    return buf.read()
