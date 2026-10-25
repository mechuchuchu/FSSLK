FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2'

COPY RL_schedule /app/RL_schedule

ENV PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
