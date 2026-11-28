FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY dashboard/requirements.txt ./requirements.txt
COPY episense_security.py ./
RUN pip install --no-cache-dir -r requirements.txt
COPY config ./config
COPY data ./data
COPY models ./models
COPY dashboard ./dashboard
RUN install -d -o root -g 1000 -m 2770 /app/auth-state
EXPOSE 8000
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
