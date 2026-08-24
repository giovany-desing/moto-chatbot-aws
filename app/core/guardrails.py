"""
Guardrails livianos y agnosticos de proveedor (punto #7 del plan de
mejora). No usamos Bedrock Guardrails porque solo aplica de forma
nativa a invocaciones vía Bedrock, y decidimos sacar Bedrock del flujo
normal -- usar el API independiente ApplyGuardrail hubiera significado
reintroducir una dependencia de Bedrock en el camino critico solo para
esto. En su lugar:

1. Enmascarado de PII antes de enviar datos a Langfuse (observabilidad
   de terceros) -- ej. un cliente puede escribir su telefono/email
   dentro del texto libre de su pregunta.
2. Deteccion heuristica de prompt injection sobre el input del usuario,
   antes de invocar al LLM -- relevante sobre todo para el asistente
   comercial, que habla con clientes externos y puede escribir en BD
   (crear_orden_servicio, registrar_lead).

Limitacion documentada: esto es una PRIMERA linea de defensa basada en
patrones/regex, no un modelo de clasificacion dedicado (ej. Llama
Guard, o un clasificador entrenado). Para produccion real a mayor
escala, valdria la pena escalar a eso -- queda como siguiente paso.
"""
import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Enmascarado de PII (para Langfuse) -------------------------------------

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Telefonos estilo Colombia: movil de 10 digitos (empieza en 3), con o
# sin prefijo +57. Se exige el patron completo (no solo "10 digitos
# seguidos") para no enmascarar por error numeros como kilometraje o
# referencias de repuestos.
_PHONE_PATTERN = re.compile(r"(?:\+?57[\s.-]?)?3\d{2}[\s.-]?\d{3}[\s.-]?\d{4}\b")


def mask_pii(texto: str) -> str:
    texto = _EMAIL_PATTERN.sub("[EMAIL_REDACTADO]", texto)
    texto = _PHONE_PATTERN.sub("[TELEFONO_REDACTADO]", texto)
    return texto


def mask_for_langfuse(*, data, **kwargs):
    """
    Funcion de enmascarado para el cliente Langfuse (ver
    app/core/observability.py). Recorre recursivamente strings, dicts
    y listas/tuplas -- cualquier otro tipo se devuelve tal cual.
    """
    if isinstance(data, str):
        return mask_pii(data)
    if isinstance(data, dict):
        return {k: mask_for_langfuse(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_for_langfuse(data=v) for v in data]
    return data


# --- Deteccion heuristica de prompt injection -------------------------------

_PATRONES_INYECCION = [
    r"ignora(?:r|s)?\s+(?:las\s+)?instrucciones",
    r"olvida(?:r|te)?\s+(?:el\s+|tu\s+)?(?:system\s+)?prompt",
    r"revela(?:r)?\s+(?:tu|el)\s+(?:prompt|instrucciones|system\s*prompt)",
    r"eres\s+(?:ahora\s+)?(?:un\s+)?(?:modelo|asistente|IA)\s+sin\s+restricciones",
    r"act[uú]a\s+como\s+(?:si\s+)?(?:no\s+tuvieras|sin)",
    r"modo\s+desarrollador",
    r"jailbreak",
    r"ignore\s+(?:previous|all)\s+instructions",
    r"disregard\s+(?:previous|all)\s+instructions",
    r"you\s+are\s+now\s+(?:in\s+)?(?:dan|developer\s+mode)",
    r"reveal\s+your\s+(?:system\s+)?prompt",
]

_REGEX_INYECCION = re.compile("|".join(_PATRONES_INYECCION), re.IGNORECASE)


def detect_prompt_injection(texto: str) -> bool:
    """
    Deteccion heuristica basada en patrones -- primera linea de defensa,
    no un clasificador entrenado. Puede tener falsos negativos (frases
    de inyeccion no cubiertas por estos patrones) y en teoria falsos
    positivos si un usuario legitimo menciona estas frases fuera de
    contexto (poco probable en este dominio de mecanica/ventas de motos).
    """
    if _REGEX_INYECCION.search(texto):
        logger.warning(f"Posible intento de prompt injection detectado: {texto[:100]!r}")
        return True
    return False
