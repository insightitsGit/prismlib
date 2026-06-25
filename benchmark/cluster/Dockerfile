FROM python:3.11-slim

WORKDIR /app

# System deps for psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY node_app.py .

EXPOSE 8080

CMD ["python", "-u", "node_app.py"]
