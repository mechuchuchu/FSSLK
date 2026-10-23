FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY fsslk ./fsslk

ENTRYPOINT ["python", "-m", "fsslk"]
