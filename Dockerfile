FROM python:3.11-slim
ARG SERVICE_VERSION=0.2.0
ARG GIT_SHA=unknown
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV SERVICE_VERSION=${SERVICE_VERSION}
ENV GIT_SHA=${GIT_SHA}
EXPOSE 8000
CMD ["uvicorn", "src.churn_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
