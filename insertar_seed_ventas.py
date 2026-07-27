import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    connect_timeout=10,
)
cur = conn.cursor()
with open("scripts/seed_ventas.sql", "r") as f:
    cur.execute(f.read())
conn.commit()
cur.execute("SELECT modelo, precio FROM catalogo_motos ORDER BY precio;")
for row in cur.fetchall():
    print(row)
conn.close()
