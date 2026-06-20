# ---------- Builder Stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY . .
RUN pip install --prefix=/install -e .

# ---------- Runtime Stage ----------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ /app/src/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "airbnb_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
