"""
Handler Lambda para el Worker de indexación — se dispara por el event
source mapping de SQS. Si un mensaje falla, se relanza para que SQS
reintente automáticamente (hasta 3 veces) y luego vaya a la DLQ.
"""
import json

from app.core.logging import get_logger
from app.workers.indexing_worker import IndexingWorker

logger = get_logger(__name__)

worker = IndexingWorker()


def handler(event, context):
    resultados = []
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            filename = body["filename"]
            resultado = worker.process(filename)
            resultados.append(resultado)
        except Exception as exc:
            logger.error(f"Error procesando mensaje SQS: {exc}")
            raise

    return {"procesados": len(resultados), "detalle": resultados}
