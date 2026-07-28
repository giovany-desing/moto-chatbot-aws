"""
5 endpoints principales de la API:
  GET  /health
  GET  /documentos
  POST /documentos
  POST /chat
  POST /chat-cliente
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.logging import get_logger
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatClienteRequest,
    ChatClienteResponse,
    DocumentInfo,
    UploadResponse,
    HealthResponse,
)
from app.services import storage_service, vector_service, rag_service, sales_service
import boto3
import json

from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

_sqs = boto3.client("sqs", region_name=settings.AWS_REGION)


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.get("/documentos", response_model=list[DocumentInfo])
def listar_documentos():
    return vector_service.list_manuales()


@router.post("/documentos", response_model=UploadResponse)
async def subir_documento(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    contenido = await file.read()
    storage_service.upload_pdf(file.filename, contenido)

    _sqs.send_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        MessageBody=json.dumps({"filename": file.filename, "bucket": settings.S3_BUCKET_NAME}),
    )

    logger.info(f"Manual {file.filename} recibido, mensaje enviado a SQS")
    return UploadResponse(
        message="Manual recibido y en proceso de indexación",
        filename=file.filename,
        status="processing",
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest)
    try:
        resultado = rag_service.answer(
            question=request.question,
            filename=request.filename,
            session_id=request.session_id,
        )
        return ChatResponse(**resultado)
    except Exception as exc:
        logger.error(f"Error procesando chat: {exc}")
        raise HTTPException(status_code=500, detail="Error procesando la consulta") from exc


@router.post("/chat-cliente", response_model=ChatClienteResponse)
def chat_cliente(request: ChatClienteRequest):
    try:
        resultado = sales_service.answer_cliente(
            question=request.question,
            session_id=request.session_id,
        )
        return ChatClienteResponse(**resultado)
    except Exception as exc:
        logger.error(f"Error procesando chat-cliente: {exc}")
        raise HTTPException(status_code=500, detail="Error procesando la consulta") from exc
