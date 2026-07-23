# ScribeSwarm — single-container image (API + web UI)
# NOTE: authored on a machine without Docker; first verified run pending.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Checkpoints + run artifacts live in /app/data (mount a volume to persist).
EXPOSE 8000
CMD ["uvicorn", "scribeswarm.api:app", "--host", "0.0.0.0", "--port", "8000"]
