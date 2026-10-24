FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2' \
    && python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple -r requirements.txt

COPY fsslk ./fsslk
COPY scripts ./scripts

ENTRYPOINT ["python", "-m", "fsslk"]
