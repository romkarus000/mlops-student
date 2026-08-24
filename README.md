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
dvc repro
python scripts/build_release_manifest.py
make verify
uvicorn src.churn_ml.api:app --reload --port 8000
```

Артефакты:

- `data/raw/churn_reference.csv` и `churn_current.csv`;
- `models/hybrid_churn_bundle.joblib`;
- `models/model_metadata.json`;
- `reports/training_metrics.json`;
- `reports/monitoring_report.html`;
- `reports/release_manifest.json` — паспорт конкретной поставки.

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

Репозиторий уже инициализирован для DVC. Первый запуск строит все артефакты и
создаёт локальный cache, повторный — пропускает неизменившиеся этапы:

```bash
dvc repro
dvc dag
dvc status
dvc metrics show
```

Этапы в `dvc.yaml`:

1. `generate_data`;
2. `train`;
3. `monitor`.

Изменение кода, `params.yaml` или исходных данных перезапускает только
зависимые этапы. `dvc.lock` фиксирует точные хеши зависимостей и результатов;
его нужно коммитить вместе с изменением pipeline. `requirements.txt` также
является зависимостью этапов, поэтому смена ML-окружения не останется незаметной.

В проекте данные синтетические, поэтому для занятия достаточно локального DVC
cache. Для общей команды подключается remote, например S3:

```bash
dvc remote add -d storage s3://your-bucket/mlops-student
dvc push
```

Адрес и учётные данные реального хранилища не коммитятся в учебный репозиторий.

## MLflow и версия модели

Каждое обучение создаёт MLflow run и сохраняет его неизменяемый ID в
`models/model_metadata.json`. Тот же ID возвращают `/health` и `/predict`.
Вместе с ним сохраняются параметры, метрики и SHA256 model bundle. Так можно
ответить не только «какой код был в Git», но и «какой именно запуск создал
модель, обслужившую запрос».

Порог классификации берётся из `params.yaml`, попадает в model bundle и
используется API. Поэтому изменение порога тоже проходит через DVC и не
расходится между обучением и инференсом.

## Паспорт релиза

После `dvc repro` выполните:

```bash
SERVICE_VERSION=0.2.0 \
GIT_SHA=$(git rev-parse HEAD) \
IMAGE_TAG=mlops-student:0.2.0 \
python scripts/build_release_manifest.py
```

`reports/release_manifest.json` связывает в одной записи:

- версию сервиса и Git SHA;
- SHA256 `dvc.lock`, `params.yaml`, полного и runtime-набора зависимостей;
- SHA256 reference/current данных;
- MLflow run ID, SHA256 модели и рабочий threshold;
- метрики и тег Docker-образа.

Скрипт завершится ошибкой, если bundle модели не совпадает с digest в
метаданных. Это простая, но реальная проверка целостности поставки.

`requirements.txt` описывает среду обучения и CI, а компактный
`requirements-runtime.txt` — только зависимости API. Поэтому Docker-образ не
содержит DVC, pytest, Evidently и другие инструменты, не нужные при инференсе.

## Реальный CI

Workflow `.github/workflows/ci.yml` на каждом pull request:

1. устанавливает зафиксированные зависимости;
2. выполняет `dvc repro`;
3. формирует паспорт релиза;
4. показывает DVC status и метрики;
5. запускает тесты и собирает Docker-образ;
6. прикладывает `dvc.lock`, метрики, metadata, monitoring report и release
   manifest как evidence к запуску GitHub Actions.

Это именно **CI**: workflow доказывает, что поставка воспроизводится и
собирается. Автоматического развёртывания здесь намеренно нет. Публикация
образа, approvals и deployment появятся на следующем шаге курса как CD.

## Метрики и ограничения

Обучение сохраняет:

- ROC AUC и Average Precision для churn;
- Accuracy и F1 для sentiment;
- размеры train/test выборок.

Данные и отзывы синтетические. Метрики показывают воспроизводимость технического
pipeline, а не готовность модели к реальному бизнес-применению. Для production
понадобятся реальные отзывы, связь с последующим фактом оттока, контроль
дисбаланса, защищённый ingress, аутентификация и постоянное хранилище мониторинга.
