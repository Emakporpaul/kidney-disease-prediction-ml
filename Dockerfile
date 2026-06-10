# Deployed as a Hugging Face Space (Docker SDK)
#
# Hugging Face Spaces require the app to listen on port 7860.
# Artefacts are downloaded from the HF model repo at startup by app.py,
# so no .pkl files are baked into the image.
#
# Build locally (optional):
#   docker build -t ckd-api .
#   docker run -p 7860:7860 ckd-api

FROM python:3.10-slim

# Keeps Python from buffering stdout/stderr — logs appear immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (cached layer — only rebuilt when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/         ./src/
COPY app.py       .
COPY schemas.py   .

# Create model/ directory — artefacts will be downloaded here at runtime
RUN mkdir -p model

# Hugging Face Spaces must listen on port 7860
EXPOSE 7860

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
