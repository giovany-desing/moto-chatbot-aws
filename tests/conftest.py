"""
Configuracion compartida de pytest. Se carga ANTES de que cualquier
test importe modulos de app/, por eso las variables de entorno se
fijan aqui como primer paso -- Settings() en app/core/config.py exige
varios campos obligatorios (DB_HOST, API_KEY, etc.) que no existen en
el entorno de CI/local de pruebas.

Estas son credenciales FALSAS, solo para que la app pueda importarse
y correr en modo unitario -- ningun test de este proyecto se conecta
a AWS, RDS o Redis reales.
"""
import os

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("S3_BUCKET_NAME", "test-bucket")
os.environ.setdefault("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue")
os.environ.setdefault("API_KEY", "test-api-key-para-pytest")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("BEDROCK_REGION", "us-east-1")
