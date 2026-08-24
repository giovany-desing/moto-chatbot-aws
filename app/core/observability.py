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
    from app.core.guardrails import mask_for_langfuse

    langfuse_client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        base_url=settings.LANGFUSE_BASE_URL,
        mask=mask_for_langfuse,
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


def get_prompt_or_fallback(name: str, fallback: str, label: str = "production") -> str:
    """
    Registro de prompts versionado (punto #10 del plan de mejora): trae
    el prompt activo (label="production" por defecto) desde Langfuse
    Cloud, permitiendo rollback o A/B testing SIN redeploy de codigo --
    solo cambiando que version tiene la etiqueta "production" en el
    dashboard de Langfuse.

    No bloqueante: si Langfuse esta deshabilitado, o el prompt no
    existe todavia, o falla la llamada por cualquier razon, cae al
    texto hardcodeado que ya tenia el codigo (mismo principio de
    degradacion segura que cache_service.py con Redis).
    """
    if langfuse_client is None:
        return fallback
    try:
        prompt_obj = langfuse_client.get_prompt(name=name, label=label, fallback=fallback, cache_ttl_seconds=300)
        return prompt_obj.prompt
    except Exception as exc:
        logger.warning(f"No se pudo obtener el prompt '{name}' de Langfuse, usando fallback local: {exc}")
        return fallback
