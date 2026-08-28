FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt awslambdaric

# Pre-descargar y guardar los modelos de embeddings/reranking como
# CARPETAS PLANAS (.save(), sin la estructura de cache de Hugging Face
# con symlinks+locks) EN TIEMPO DE BUILD.
#
# HISTORIAL REAL: el primer intento horneaba la cache de HF tal cual
# (snapshot_download) y copiaba esa cache a /tmp en runtime porque
# /var/task es de solo lectura en Lambda -- pero la cache de HF tiene
# miles de archivos pequeños/symlinks, y copiarla a /tmp tardaba mas de
# 120 segundos en Lambda real (confirmado con CloudWatch logs). La
# cache de HF tambien intenta escribir locks incluso con
# HF_HUB_OFFLINE=1, colgandose contra un filesystem de solo lectura.
#
# Fix real: guardar cada modelo como carpeta plana autocontenida
# (SentenceTransformer/CrossEncoder .save()) y cargar DIRECTAMENTE desde
# esa ruta en runtime -- sin pasar por la logica de cache/resolucion de
# Hugging Face en absoluto, sin intentos de escritura, sin copia a /tmp.
# Verificado con una prueba real (chmod -R a-w, solo lectura de verdad)
# que ambos modelos cargan y funcionan sin ningun intento de escritura.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3', device='cpu').save('/var/task/models/bge-m3'); \
CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu').save('/var/task/models/bge-reranker-v2-m3')" \
    && rm -rf /root/.cache/huggingface

ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY lambdas ./lambdas

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambdas.api_handler.handler"]
