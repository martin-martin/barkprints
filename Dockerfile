FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Only the runtime deps of the web app. sentence-transformers/torch are used
# solely by the offline corpus_builder, never at request time, so they are
# deliberately excluded to keep the image small.
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    scipy

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "barkprints.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
