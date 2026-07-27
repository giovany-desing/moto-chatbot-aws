"""
Herramientas MCP de atención al cliente / ventas — el "empleado digital"
comercial. Consulta catálogo, simula financiamiento y escala a un asesor
humano cuando el caso lo requiere (ver leads).
"""
from app.core.logging import get_logger
from app.services.db_service import db_cursor

logger = get_logger(__name__)


def buscar_motos_catalogo(uso: str | None = None, presupuesto_max: float | None = None) -> dict:
    """Busca motos en el catálogo filtrando por uso recomendado y/o presupuesto máximo."""
    condiciones = ["disponible = TRUE"]
    parametros: list = []

    if uso:
        condiciones.append("%s = ANY(uso_recomendado)")
        parametros.append(uso)
    if presupuesto_max:
        condiciones.append("precio <= %s")
        parametros.append(presupuesto_max)

    where_clause = " AND ".join(condiciones)

    with db_cursor() as (conn, cur):
        cur.execute(
            f"""
            SELECT modelo, marca, cilindraje, precio, uso_recomendado, descripcion
            FROM catalogo_motos
            WHERE {where_clause}
            ORDER BY precio ASC
            """,
            parametros,
        )
        rows = cur.fetchall()

    return {
        "encontradas": len(rows),
        "motos": [
            {
                "modelo": r[0],
                "marca": r[1],
                "cilindraje": r[2],
                "precio": float(r[3]),
                "uso_recomendado": r[4],
                "descripcion": r[5],
            }
            for r in rows
        ],
    }


def recomendar_moto(necesidad: str, presupuesto_max: float | None = None) -> dict:
    """
    Recomienda una o varias motos según la necesidad expresada por el
    cliente en lenguaje natural (ej. "para trabajar en domicilios",
    "para viajar en carretera", "algo económico para la ciudad").
    Usa buscar_motos_catalogo internamente con un mapeo simple de
    palabras clave a categorías de uso_recomendado.
    """
    necesidad_lower = necesidad.lower()
    mapeo = {
        "urbano": ["ciudad", "urbano", "diario", "trabajo"],
        "economico": ["economic", "barato", "bajo consumo", "ahorro"],
        "domicilios": ["domicilio", "reparto", "delivery"],
        "deportivo": ["deportiv", "rapida", "rápida", "potencia"],
        "touring": ["viaje", "carretera", "turismo", "largo"],
    }

    categoria_detectada = None
    for categoria, palabras_clave in mapeo.items():
        if any(palabra in necesidad_lower for palabra in palabras_clave):
            categoria_detectada = categoria
            break

    resultado = buscar_motos_catalogo(uso=categoria_detectada, presupuesto_max=presupuesto_max)

    return {
        "categoria_detectada": categoria_detectada or "general",
        "recomendaciones": resultado["motos"][:3],
    }


def simular_financiamiento(precio_moto: float, entrada: float, plazo_meses: int) -> dict:
    """
    Calcula una cuota mensual estimada (sistema de amortización francés)
    para cada opción de financiamiento activa. Es una ESTIMACIÓN
    REFERENCIAL, no una oferta vinculante — así se le indica al cliente
    en la respuesta del asistente (ver system prompt en sales_service.py).
    """
    monto_financiar = precio_moto - entrada
    if monto_financiar <= 0:
        return {"error": "La entrada cubre el valor total de la moto, no se requiere financiamiento"}

    with db_cursor() as (conn, cur):
        cur.execute(
            """
            SELECT entidad, tasa_interes_mensual, plazo_max_meses, entrada_minima_pct
            FROM financiamiento_opciones
            WHERE activo = TRUE AND plazo_max_meses >= %s
            ORDER BY tasa_interes_mensual ASC
            """,
            [plazo_meses],
        )
        rows = cur.fetchall()

    entrada_pct = (entrada / precio_moto) * 100
    opciones = []
    for entidad, tasa, plazo_max, entrada_minima_pct in rows:
        tasa = float(tasa)
        if entrada_pct < float(entrada_minima_pct):
            continue
        cuota = monto_financiar * (tasa * (1 + tasa) ** plazo_meses) / (((1 + tasa) ** plazo_meses) - 1)
        opciones.append({
            "entidad": entidad,
            "cuota_mensual_estimada": round(cuota, 2),
            "plazo_meses": plazo_meses,
            "tasa_interes_mensual": tasa,
        })

    return {
        "precio_moto": precio_moto,
        "entrada": entrada,
        "monto_financiar": monto_financiar,
        "es_estimacion_referencial": True,
        "opciones": opciones,
    }


def registrar_lead(nombre_cliente: str, interes: str, telefono: str | None = None, email: str | None = None, motivo_escalamiento: str | None = None, session_id: str | None = None) -> dict:
    """
    Registra un lead para que un asesor humano dé seguimiento. Úsala
    cuando el cliente pida algo que el asistente no puede resolver solo
    (negociar precio, trámite formal de crédito, prueba de manejo, etc.)
    o cuando explícitamente pida hablar con un asesor.
    """
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            INSERT INTO leads (nombre_cliente, telefono, email, interes, motivo_escalamiento, session_id, estado)
            VALUES (%s, %s, %s, %s, %s, %s, 'nuevo')
            RETURNING id
            """,
            [nombre_cliente, telefono, email, interes, motivo_escalamiento, session_id],
        )
        lead_id = cur.fetchone()[0]

    return {"lead_id": lead_id, "estado": "registrado", "mensaje": "Un asesor se pondrá en contacto pronto"}


_TOOLS = {
    "buscar_motos_catalogo": buscar_motos_catalogo,
    "recomendar_moto": recomendar_moto,
    "simular_financiamiento": simular_financiamiento,
    "registrar_lead": registrar_lead,
}

_TOOLS_SCHEMA = [
    {
        "toolSpec": {
            "name": "buscar_motos_catalogo",
            "description": "Busca motos disponibles en el catálogo, opcionalmente filtrando por tipo de uso y presupuesto máximo. Úsala cuando el cliente pregunte qué motos hay disponibles.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "uso": {"type": "string", "description": "Categoría de uso: urbano, economico, domicilios, deportivo o touring"},
                        "presupuesto_max": {"type": "number"},
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "recomendar_moto",
            "description": "Recomienda motos según la necesidad expresada por el cliente en lenguaje natural. Úsala cuando el cliente describa para qué la necesita en vez de pedir un modelo específico.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "necesidad": {"type": "string"},
                        "presupuesto_max": {"type": "number"},
                    },
                    "required": ["necesidad"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "simular_financiamiento",
            "description": "Calcula una cuota mensual estimada de financiamiento para una moto. Úsala cuando el cliente pregunte por cuotas, crédito o financiamiento. SIEMPRE aclara al cliente que es una estimación referencial, no una oferta formal.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "precio_moto": {"type": "number"},
                        "entrada": {"type": "number"},
                        "plazo_meses": {"type": "integer"},
                    },
                    "required": ["precio_moto", "entrada", "plazo_meses"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "registrar_lead",
            "description": "Registra los datos del cliente para que un asesor humano lo contacte. Úsala cuando el cliente pida hablar con un asesor, o cuando la solicitud requiera algo que el asistente no puede resolver solo (negociación, trámite formal, prueba de manejo).",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "nombre_cliente": {"type": "string"},
                        "interes": {"type": "string"},
                        "telefono": {"type": "string"},
                        "email": {"type": "string"},
                        "motivo_escalamiento": {"type": "string"},
                    },
                    "required": ["nombre_cliente", "interes"],
                }
            },
        }
    },
]


def get_tools_schema() -> list[dict]:
    return _TOOLS_SCHEMA


def execute_tool(name: str, parameters: dict) -> dict:
    if name not in _TOOLS:
        logger.warning(f"Herramienta MCP de ventas desconocida solicitada: {name}")
        return {"error": f"herramienta '{name}' no existe"}
    try:
        return _TOOLS[name](**parameters)
    except Exception as exc:
        logger.error(f"Error ejecutando herramienta de ventas {name}: {exc}")
        return {"error": str(exc)}
