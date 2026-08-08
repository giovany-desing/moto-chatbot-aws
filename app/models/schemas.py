"""
Schemas Pydantic v2 para requests/responses de la API.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Pregunta del mecánico")
    filename: str | None = Field(default=None, description="Filtrar búsqueda a un manual específico")
    session_id: str | None = Field(default=None, description="Activa memoria conversacional")


class SourceChunk(BaseModel):
    filename: str
    page: int
    relevance: float
    text_preview: str | None = Field(default=None, description="Fragmento corto del chunk fuente, util para citar/evaluar")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    from_cache: bool = False
    tools_used: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int
    indexed: bool
    created_at: datetime


class UploadResponse(BaseModel):
    message: str
    filename: str
    status: str


class ChatClienteRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Pregunta del cliente")
    session_id: str | None = Field(default=None, description="Activa memoria conversacional")


class ChatClienteResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    from_cache: bool = False
    tools_used: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str = "moto-chatbot"
