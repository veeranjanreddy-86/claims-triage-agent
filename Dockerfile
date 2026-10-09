FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLAIMS_DB_PATH=/app/var/claims.db \
    CLAIMS_TRACE_DIR=/app/traces

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && mkdir -p /app/var /app/traces && chown -R app:app /app

USER app
RUN python -m claims_agent.data.seed --db /app/var/claims.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "claims_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
