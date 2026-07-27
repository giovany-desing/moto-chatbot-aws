"""
Modelos SQLAlchemy — PENDIENTE.

Actualmente las tablas se crean con SQL crudo en app/services/db_service.py
(CREATE TABLE IF NOT EXISTS en cada cold start). Este módulo queda como
punto de entrada para migrar a SQLAlchemy ORM + Alembic cuando el
esquema se estabilice.
"""
