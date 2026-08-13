"""
Autenticacion por API Key a nivel de aplicacion.

El API Gateway de este proyecto es tipo HTTP API (no REST API), que NO
soporta el sistema nativo de API Keys/Usage Plans de AWS (eso es
exclusivo de REST API v1). Por eso la validacion se hace aqui, dentro
de FastAPI, como una dependencia que se aplica a cada endpoint sensible.

El cliente debe mandar el header:
    X-API-Key: <la clave>

/health queda deliberadamente SIN esta proteccion -- es informacion no
sensible, util para monitoreo externo sin necesitar credenciales.
"""
import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Dependencia de FastAPI. Usar como: Depends(verify_api_key) en cada
    ruta que deba protegerse. Compara con secrets.compare_digest para
    evitar timing attacks (que alguien deduzca la clave midiendo cuanto
    tarda la comparacion caracter por caracter).
    """
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        logger.warning("Intento de acceso con API Key invalida o ausente")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key invalida o ausente. Incluye el header X-API-Key.",
        )
