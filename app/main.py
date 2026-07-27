"""
Punto de entrada FastAPI. Ejecuta migraciones de DB en cada cold start.
Mangum adapta la app ASGI al formato de eventos de Lambda + API Gateway.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.api.middleware import add_cors, RequestLoggingMiddleware
from app.core.logging import get_logger
from app.services.db_service import run_migrations

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        run_migrations()
    except Exception as exc:
        logger.error(f"No se pudieron aplicar migraciones en el arranque: {exc}")
    yield


app = FastAPI(
    title="Moto Chatbot API",
    description="Chatbot técnico para talleres de motocicletas — RAG sobre manuales de taller",
    version="1.0.0",
    lifespan=lifespan,
)

add_cors(app)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(router)
