FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt awslambdaric

# Pre-descargar (hornear) los modelos de embeddings/reranking EN TIEMPO
# DE BUILD -- sin esto, cada cold start de Lambda tendria que descargar
# los modelos desde Hugging Face antes de poder responder la primera
# peticion. Se excluyen imagenes/assets/onnx/*.pt (no se usan, solo
# ocupan espacio) -- NO se puede filtrar por "solo safetensors": bge-m3
# unicamente tiene pytorch_model.bin (sin variante safetensors) y
# bge-reranker-v2-m3 es al reves (solo safetensors, sin .bin) --
# confirmado consultando el listado real de archivos de cada repo.
ENV HF_HOME=/var/task/.hf_cache
RUN python -c "\
from huggingface_hub import snapshot_download; \
excluir = ['imgs/*', 'assets/*', 'onnx/*', '*.pt', '*.jpg', '*.webp', '*.png', '.DS_Store']; \
snapshot_download(repo_id='BAAI/bge-m3', ignore_patterns=excluir); \
snapshot_download(repo_id='BAAI/bge-reranker-v2-m3', ignore_patterns=excluir)"

# En runtime, NUNCA intentar red para cargar los modelos -- usar solo lo
# horneado en la imagen. El filesystem de Lambda es de solo lectura
# excepto /tmp, y esto tambien evita fallos si Hugging Face no responde
# en ese momento.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY lambdas ./lambdas

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambdas.api_handler.handler"]
