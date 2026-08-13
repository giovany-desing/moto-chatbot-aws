"""
Tests de degradacion segura de Redis (app/services/cache_service.py).
Confirma que si Redis falla, el sistema NO se cae -- degrada a un
estado seguro (sin cache, sin memoria) en vez de propagar la excepcion.
"""
import redis
import pytest

from app.services import cache_service


class _ClienteRedisQueFalla:
    """Simula un cliente Redis que siempre lanza RedisError -- sin
    necesitar un servidor Redis real corriendo durante los tests."""

    def get(self, *args, **kwargs):
        raise redis.RedisError("Redis no disponible (simulado)")

    def setex(self, *args, **kwargs):
        raise redis.RedisError("Redis no disponible (simulado)")


@pytest.fixture(autouse=True)
def _reset_cliente_redis(monkeypatch):
    # Evita que un cliente cacheado de un test anterior contamine el siguiente
    monkeypatch.setattr(cache_service, "_redis_client", None)


def test_get_degrada_a_none_si_redis_falla(mocker):
    mocker.patch.object(cache_service, "get_client", return_value=_ClienteRedisQueFalla())
    resultado = cache_service.get("¿cada cuanto se cambia el aceite?")
    assert resultado is None


def test_set_no_lanza_excepcion_si_redis_falla(mocker):
    mocker.patch.object(cache_service, "get_client", return_value=_ClienteRedisQueFalla())
    cache_service.set("pregunta", {"answer": "respuesta"})  # no debe lanzar nada


def test_get_memory_degrada_a_lista_vacia_si_redis_falla(mocker):
    mocker.patch.object(cache_service, "get_client", return_value=_ClienteRedisQueFalla())
    resultado = cache_service.get_memory("sesion-1")
    assert resultado == []


def test_save_memory_no_lanza_excepcion_si_redis_falla(mocker):
    mocker.patch.object(cache_service, "get_client", return_value=_ClienteRedisQueFalla())
    cache_service.save_memory("sesion-1", "pregunta", "respuesta")  # no debe lanzar nada


def test_get_memory_sin_session_id_no_toca_redis(mocker):
    get_client_mock = mocker.patch.object(cache_service, "get_client")
    resultado = cache_service.get_memory("")
    assert resultado == []
    get_client_mock.assert_not_called()
