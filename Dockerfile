FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt awslambdaric

COPY app ./app
COPY lambdas ./lambdas

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambdas.api_handler.handler"]
