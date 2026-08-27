"""
Copia el cache de Hugging Face (horneado en /var/task/.hf_cache durante
el build de Docker, ver Dockerfile) a /tmp en tiempo de ejecucion, SOLO
dentro de Lambda.

Motivo real: /var/task es de SOLO LECTURA en Lambda -- aunque
HF_HUB_OFFLINE=1 evita llamadas de red, la libreria de Hugging Face
igual intenta operaciones de escritura (archivos .lock, verificacion de
cache) al cargar un modelo, y esas escrituras se cuelgan/fallan en
silencio contra un filesystem de solo lectura. /tmp SI es escribible
(hasta el tamaño configurado en EphemeralStorage).

Se copia UNA SOLA VEZ por contenedor (Lambda puede reusar el mismo
contenedor "caliente" entre invocaciones -- no repetir la copia en cada
request).
"""
import os
import shutil

from app.core.logging import get_logger

logger = get_logger(__name__)

_HF_CACHE_HORNEADO = "/var/task/.hf_cache"
_HF_CACHE_ESCRIBIBLE = "/tmp/.hf_cache"

_ya_copiado = False


def ensure_writable_hf_cache() -> None:
    """
    Si estamos en Lambda (AWS_LAMBDA_FUNCTION_NAME presente) y el cache
    horneado existe, lo copia a /tmp (escribible) y apunta HF_HOME ahi
    -- debe llamarse ANTES de importar sentence_transformers/torch por
    primera vez, para que la variable de entorno tenga efecto.
    En local (fuera de Lambda) no hace nada -- se sigue usando el
    HF_HOME normal del sistema.
    """
    global _ya_copiado

    en_lambda = "AWS_LAMBDA_FUNCTION_NAME" in os.environ
    if not en_lambda:
        return

    if _ya_copiado:
        return

    if not os.path.isdir(_HF_CACHE_HORNEADO):
        logger.warning(f"No se encontro cache horneado en {_HF_CACHE_HORNEADO}, se omite la copia")
        _ya_copiado = True
        return

    if not os.path.isdir(_HF_CACHE_ESCRIBIBLE):
        logger.info(f"Copiando cache de modelos de {_HF_CACHE_HORNEADO} a {_HF_CACHE_ESCRIBIBLE} (solo una vez por contenedor)")
        shutil.copytree(_HF_CACHE_HORNEADO, _HF_CACHE_ESCRIBIBLE)
        logger.info("✅ Cache copiado a /tmp")

    os.environ["HF_HOME"] = _HF_CACHE_ESCRIBIBLE
    _ya_copiado = True
