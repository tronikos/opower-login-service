FROM python:3.13-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && \
    apt-get upgrade -y && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    playwright install --with-deps firefox && \
    apt-get clean && \
    rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*
COPY app/ .

VOLUME "/data"
ENV SESSION_STORAGE_PATH=/data

EXPOSE 7937
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7937"]
