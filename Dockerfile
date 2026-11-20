FROM python:3.11-slim

# LightGBM needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir "lightgbm>=4.5" "pandas>=2.2" "numpy>=1.26" "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7"

COPY src/ src/
COPY api/ api/
COPY models/ models/

# Cloud Run injects $PORT (default 8080)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
