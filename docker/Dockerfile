FROM tensorflow/tensorflow:2.10.0-gpu

RUN apt-get update && \
    apt-get install -y mongodb-clients && \
    apt-get clean

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt && \
    mkdir -p /app/models && \
    chmod +x /app/scripts/*.sh

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "api.routes:retrain_bp"]