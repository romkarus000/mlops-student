FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
EXPOSE 8000
CMD ["uvicorn", "src.churn_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
