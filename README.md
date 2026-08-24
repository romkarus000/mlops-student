# Учебный MLOps-проект: гибридный прогноз оттока

Проект показывает полный MLOps-цикл на задаче прогноза оттока клиентов. API
анализирует текст отзыва, объединяет вероятность негативной тональности с
поведенческими признаками клиента и возвращает риск оттока.

Негативный отзыв повышает риск, но не определяет его автоматически: churn-модель
также учитывает активность, срок жизни клиента, тариф, обращения в поддержку,
задержки оплаты и NPS.

## Архитектура

| Компонент | Роль |
|---|---|
| `scripts/generate_data.py` | создаёт синтетические reference/current данные, отзывы и метки тональности |
| TF-IDF + Logistic Regression | определяет positive/negative sentiment отзыва |
| Churn Logistic Regression | использует клиентские признаки и negative sentiment probability |
| MLflow | хранит параметры, метрики, bundle модели и неизменяемый run ID |
| FastAPI | предоставляет `/health` и `/predict` с трассировкой запросов |
| Evidently | формирует отчёт о data drift |
| DVC | описывает воспроизводимый pipeline data → train → monitor |
| Docker Compose | запускает MLflow и API |
| GitHub Actions | проверяет pipeline, тесты, мониторинг и Docker-сборку |

## Контракт API

Swagger UI после запуска доступен на `http://localhost:8000/docs`.

Пример запроса:

```json
{
  "customer_id": "C-HIGH-RISK",
  "review_text": "Разочарован качеством, хочу отменить подписку",
  "customer": {
    "tenure_months": 2,
    "monthly_fee": 1990,
    "days_active_last_30": 2,
    "support_tickets_last_30": 6,
    "payment_delay_days": 8,
    "nps_score": 1,
    "plan": "premium"
  }
}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-request-001" \
  --data-binary @examples/predict_customer.json
```

Пример ответа:

```json
{
  "request_id": "demo-request-001",
  "customer_id": "C-HIGH-RISK",
  "service": {
    "version": "0.2.0",
    "git_sha": "abc1234"
  },
  "model": {
    "name": "hybrid-review-churn",
    "version": "mlflow-run-id",
    "mlflow_run_id": "mlflow-run-id",
    "sha256": "model-digest",
    "trained_at": "2026-08-24T12:00:00+00:00"
  },
  "review_analysis": {
    "sentiment": "negative",
    "negative_probability": 0.97
  },
  "churn_prediction": {
    "probability": 0.63,
    "risk": "high",
    "threshold": 0.5
  }
}
```

Если `X-Request-ID` не передан, API создаёт UUID. Идентификатор возвращается
одновременно в JSON и HTTP-заголовке. Версия модели — MLflow run ID, SHA256
позволяет проверить целостность конкретного bundle.

## Локальный запуск

Нужен Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_data.py
python -m src.churn_ml.training
python -m src.churn_ml.monitoring
python -m pytest -q
uvicorn src.churn_ml.api:app --reload --port 8000
```

Артефакты:

- `data/raw/churn_reference.csv` и `churn_current.csv`;
- `models/hybrid_churn_bundle.joblib`;
- `models/model_metadata.json`;
- `reports/training_metrics.json`;
- `reports/monitoring_report.html`.

Без переменной `MLFLOW_TRACKING_URI` обучение использует локальный каталог
`mlruns`. При запуске через Compose run записывается в MLflow-сервис.

## Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

Сервисы:

- API: `http://localhost:8000`;
- Swagger: `http://localhost:8000/docs`;
- MLflow: `http://localhost:5000`.

Для корректных версий образ можно собрать так:

```bash
SERVICE_VERSION=0.2.0 GIT_SHA=abc1234 docker compose build api
```

## DVC pipeline

```bash
dvc init
dvc repro
```

Этапы в `dvc.yaml`:

1. `generate_data`;
2. `train`;
3. `monitor`.

Изменение кода, `params.yaml` или исходных данных перезапускает только
зависимые этапы.

## Метрики и ограничения

Обучение сохраняет:

- ROC AUC и Average Precision для churn;
- Accuracy и F1 для sentiment;
- размеры train/test выборок.

Данные и отзывы синтетические. Метрики показывают воспроизводимость технического
pipeline, а не готовность модели к реальному бизнес-применению. Для production
понадобятся реальные отзывы, связь с последующим фактом оттока, контроль
дисбаланса, защищённый ingress, аутентификация и постоянное хранилище мониторинга.
