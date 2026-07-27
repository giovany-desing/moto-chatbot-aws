-- Datos de ejemplo para catálogo y financiamiento.
-- Reemplaza estos valores por el catálogo real de tu empresa.

INSERT INTO catalogo_motos (modelo, marca, cilindraje, precio, uso_recomendado, descripcion, disponible)
VALUES
    ('Pulsar 135LS', 'Bajaj', 135, 6890000, ARRAY['urbano','economico'], 'Moto urbana ágil, ideal para ciudad y bajo consumo.', TRUE),
    ('Pulsar NS200', 'Bajaj', 200, 10990000, ARRAY['deportivo','urbano'], 'Deportiva de entrada, buen equilibrio potencia/precio.', TRUE),
    ('Dominar 400', 'Bajaj', 373, 16990000, ARRAY['touring','carretera'], 'Para viajes largos y carretera, mayor torque.', TRUE),
    ('Discover 125', 'Bajaj', 125, 5490000, ARRAY['urbano','economico','domicilios'], 'Bajo cilindraje, ideal para trabajo diario y domicilios.', TRUE)
ON CONFLICT DO NOTHING;

INSERT INTO financiamiento_opciones (entidad, tasa_interes_mensual, plazo_max_meses, entrada_minima_pct, activo)
VALUES
    ('Banco Aliado A', 0.0195, 36, 20.00, TRUE),
    ('Banco Aliado B', 0.0220, 48, 10.00, TRUE),
    ('Financiera de la empresa', 0.0250, 24, 30.00, TRUE)
ON CONFLICT DO NOTHING;
