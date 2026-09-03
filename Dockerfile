FROM python:3.11-slim
ARG SERVICE_VERSION=0.2.0
ARG GIT_SHA=unknown
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY . .
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV SERVICE_VERSION=${SERVICE_VERSION}
ENV GIT_SHA=${GIT_SHA}
# mlflow's own defaults are 120s timeout x 7 retries with backoff - fine for
# a training script, but /health calls this on every readiness probe, so an
# unreachable registry must fail fast instead of hanging the whole service
ENV MLFLOW_HTTP_REQUEST_TIMEOUT=5
ENV MLFLOW_HTTP_REQUEST_MAX_RETRIES=1
EXPOSE 8000
CMD ["uvicorn", "src.churn_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
