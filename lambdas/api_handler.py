"""
Handler Lambda para la API — adapta FastAPI a eventos de API Gateway HTTP
API vía Mangum. api_gateway_base_path="/dev" elimina el prefijo del stage.
"""
from mangum import Mangum

from app.main import app

handler = Mangum(app, api_gateway_base_path="/dev")
