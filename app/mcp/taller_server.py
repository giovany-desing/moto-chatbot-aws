"""
Herramientas MCP del taller — evolución del chatbot de "consultor de
manuales" a "asistente operativo completo".

Cada tool tiene un modelo Pydantic de validacion en execute_tool(), que
se aplica ANTES de llamar a la funcion real -- el LLM puede pedir
cualquier valor via tool-calling, y el inputSchema/toolSpec que ve el
modelo es solo una sugerencia declarativa, no una garantia. Sin esta
validacion, un kilometraje negativo o una lista de trabajos vacia
pasaban directo al INSERT sin fricción.
"""
from pydantic import BaseModel, Field, ValidationError

from app.core.logging import get_logger
from app.services.db_service import db_cursor

logger = get_logger(__name__)


def consultar_inventario(repuesto: str) -> dict:
    """
    Consulta el stock de un repuesto en el inventario del taller.

    Usa similitud por trigramas (pg_trgm) ademas de ILIKE, para tolerar
    plural/singular y variaciones menores (ej. "pastillas de freno" SI
    encuentra "Pastilla de freno delantera" -- con ILIKE puro no
    coincidian). Tambien busca contra moto_compatible, para permitir
    preguntar por el modelo de moto en vez del nombre exacto del repuesto.
    """
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            SELECT nombre, referencia, stock, precio, ubicacion,
                   GREATEST(
                       similarity(nombre, %s),
                       similarity(referencia, %s),
                       COALESCE((SELECT MAX(similarity(m, %s)) FROM unnest(moto_compatible) AS m), 0)
                   ) AS relevancia
            FROM inventario
            WHERE nombre ILIKE %s
               OR referencia ILIKE %s
               OR similarity(nombre, %s) > 0.25
               OR similarity(referencia, %s) > 0.25
               OR EXISTS (SELECT 1 FROM unnest(moto_compatible) AS m WHERE similarity(m, %s) > 0.25)
            ORDER BY relevancia DESC, stock DESC
            """,
            [repuesto, repuesto, repuesto, f"%{repuesto}%", f"%{repuesto}%", repuesto, repuesto, repuesto],
        )
        rows = cur.fetchall()

    return {
        "encontrados": len(rows),
        "repuestos": [
            {"nombre": r[0], "referencia": r[1], "stock": r[2], "precio": float(r[3]) if r[3] else None, "ubicacion": r[4]}
            for r in rows
        ],
    }


def crear_orden_servicio(cliente: str, moto: str, kilometraje: int, trabajos: list[str], placa: str | None = None) -> dict:
    """Crea una orden de servicio en el sistema del taller."""
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            INSERT INTO ordenes_servicio (cliente, moto, placa, kilometraje, trabajos, estado, fecha)
            VALUES (%s, %s, %s, %s, %s, 'abierta', NOW())
            RETURNING id
            """,
            [cliente, moto, placa, kilometraje, trabajos],
        )
        orden_id = cur.fetchone()[0]

    return {"orden_id": orden_id, "estado": "creada", "mensaje": f"Orden #{orden_id} creada"}


def historial_vehiculo(placa: str) -> dict:
    """Consulta el historial de mantenimientos de un vehículo por placa."""
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            SELECT fecha, kilometraje, trabajos, mecanico, estado
            FROM ordenes_servicio
            WHERE placa ILIKE %s
            ORDER BY fecha DESC
            LIMIT 10
            """,
            [f"%{placa}%"],
        )
        rows = cur.fetchall()

    return {
        "placa": placa,
        "total_servicios": len(rows),
        "historial": [
            {"fecha": str(r[0]), "kilometraje": r[1], "trabajos": r[2], "mecanico": r[3], "estado": r[4]}
            for r in rows
        ],
    }


def verificar_repuestos_mantenimiento(repuestos_necesarios: list[str]) -> dict:
    """
    Verifica disponibilidad de una lista de repuestos (obtenida previamente
    del manual vía RAG) contra el inventario real del taller.
    """
    resultado = []
    for repuesto in repuestos_necesarios:
        stock = consultar_inventario(repuesto)
        disponible = stock["encontrados"] > 0 and stock["repuestos"][0]["stock"] > 0
        resultado.append({
            "repuesto": repuesto,
            "disponible": disponible,
            "stock": stock["repuestos"][0]["stock"] if stock["encontrados"] > 0 else 0,
        })

    return {
        "puede_realizar": all(r["disponible"] for r in resultado),
        "repuestos": resultado,
        "faltantes": [r["repuesto"] for r in resultado if not r["disponible"]],
    }


_TOOLS = {
    "consultar_inventario": consultar_inventario,
    "crear_orden_servicio": crear_orden_servicio,
    "historial_vehiculo": historial_vehiculo,
    "verificar_repuestos_mantenimiento": verificar_repuestos_mantenimiento,
}

_TOOLS_SCHEMA = [
    {
        "toolSpec": {
            "name": "consultar_inventario",
            "description": "Consulta el stock de un repuesto en el inventario del taller. Úsala cuando el mecánico pregunte por disponibilidad de repuestos o cuando el manual indique qué repuestos se necesitan.",
            "inputSchema": {"json": {"type": "object", "properties": {"repuesto": {"type": "string"}}, "required": ["repuesto"]}},
        }
    },
    {
        "toolSpec": {
            "name": "crear_orden_servicio",
            "description": "Crea una orden de servicio en el sistema del taller. Úsala cuando el mecánico quiera registrar un trabajo a realizar.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cliente": {"type": "string"},
                        "moto": {"type": "string"},
                        "kilometraje": {"type": "integer"},
                        "trabajos": {"type": "array", "items": {"type": "string"}},
                        "placa": {"type": "string"},
                    },
                    "required": ["cliente", "moto", "kilometraje", "trabajos"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "historial_vehiculo",
            "description": "Consulta el historial completo de mantenimientos de un vehículo. Úsala cuando el mecánico mencione una placa o quiera saber qué trabajos se le han hecho a una moto.",
            "inputSchema": {"json": {"type": "object", "properties": {"placa": {"type": "string"}}, "required": ["placa"]}},
        }
    },
    {
        "toolSpec": {
            "name": "verificar_repuestos_mantenimiento",
            "description": "Verifica si el taller tiene todos los repuestos necesarios para un mantenimiento, cruzando la lista extraída del manual con el inventario real.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"repuestos_necesarios": {"type": "array", "items": {"type": "string"}}},
                    "required": ["repuestos_necesarios"],
                }
            },
        }
    },
]


def get_tools_schema() -> list[dict]:
    return _TOOLS_SCHEMA


# =====================================================================
# Validacion (Pydantic) -- se aplica ANTES de ejecutar cada tool, ver
# execute_tool(). El inputSchema de arriba es solo lo que el LLM ve
# como sugerencia; esto es lo que realmente se hace cumplir.
# =====================================================================

class _ConsultarInventarioParams(BaseModel):
    repuesto: str = Field(min_length=1, max_length=200)


class _CrearOrdenServicioParams(BaseModel):
    cliente: str = Field(min_length=1, max_length=200)
    moto: str = Field(min_length=1, max_length=200)
    kilometraje: int = Field(ge=0, le=500_000)
    trabajos: list[str] = Field(min_length=1)
    placa: str | None = Field(default=None, max_length=20)


class _HistorialVehiculoParams(BaseModel):
    placa: str = Field(min_length=1, max_length=20)


class _VerificarRepuestosParams(BaseModel):
    repuestos_necesarios: list[str] = Field(min_length=1)


_VALIDACION = {
    "consultar_inventario": _ConsultarInventarioParams,
    "crear_orden_servicio": _CrearOrdenServicioParams,
    "historial_vehiculo": _HistorialVehiculoParams,
    "verificar_repuestos_mantenimiento": _VerificarRepuestosParams,
}


def execute_tool(name: str, parameters: dict) -> dict:
    if name not in _TOOLS:
        logger.warning(f"Herramienta MCP desconocida solicitada: {name}")
        return {"error": f"herramienta '{name}' no existe"}

    modelo_validacion = _VALIDACION.get(name)
    if modelo_validacion:
        try:
            parametros_validados = modelo_validacion(**parameters).model_dump()
        except ValidationError as exc:
            logger.warning(f"Parametros invalidos para tool {name}: {exc}")
            return {"error": f"parametros invalidos para '{name}': {exc.errors()[0]['msg']}"}
    else:
        parametros_validados = parameters

    try:
        return _TOOLS[name](**parametros_validados)
    except Exception as exc:
        logger.error(f"Error ejecutando herramienta {name}: {exc}")
        return {"error": str(exc)}
