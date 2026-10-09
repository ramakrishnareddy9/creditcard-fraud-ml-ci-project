
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=30 --retries=2 \
    numpy pandas scikit-learn joblib Flask

COPY app.py .
COPY artifacts/fraud_detection_model.pkl artifacts/fraud_detection_model.pkl

EXPOSE 5000

CMD ["python", "app.py"]
