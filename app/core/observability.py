"""
Observabilidad de trazas via Langfuse Cloud (punto #4 del plan de mejora).

No bloqueante: si LANGFUSE_SECRET_KEY/LANGFUSE_PUBLIC_KEY no estan
configuradas, @observe se vuelve un no-op transparente -- el pipeline
funciona identico sin trazas, nunca depende de Langfuse para operar
(mismo principio de degradacion segura que cache_service.py con Redis).

Uso: decorar funciones con @observe() crea automaticamente un span
anidado dentro de la traza activa -- si rag_service.answer() (decorada)
llama a vector_service.search_hybrid() (tambien decorada), Langfuse
arma el arbol completo de la request sin configuracion adicional.
"""
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

langfuse_client = None

if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    from langfuse import Langfuse, observe as _observe_real

    langfuse_client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        base_url=settings.LANGFUSE_BASE_URL,
    )
    observe = _observe_real
    logger.info("Langfuse habilitado -- las trazas se enviaran a Langfuse Cloud")
else:
    def observe(*args, **kwargs):
        """No-op: reemplaza @observe cuando Langfuse no esta configurado."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    logger.info("Langfuse deshabilitado (faltan API keys) -- el pipeline corre sin trazas")
