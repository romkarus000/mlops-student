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
| MLflow Model Registry | версионирует модель (`v1`, `v2`, ...) и хранит alias `production` |
| FastAPI | предоставляет `/health`, `/predict`, `/admin/reload-model` с трассировкой запросов |
| Evidently | формирует отчёт о data drift |
| DVC | описывает воспроизводимый pipeline data → train → monitor |
| Docker Compose | запускает MLflow и API |
| GitHub Actions (CI) | `ci.yml` — проверяет pipeline, тесты, мониторинг и Docker-сборку |
| GitHub Actions (CD) | `cd.yml` — публикует образ в GHCR, деплоит через Docker Compose и раскатывает `churn-api` в Kubernetes (k3s) |
| Kubernetes (k3s) | `k8s/deployment.yaml`+`k8s/service.yaml` — 2 реплики `churn-api` за `Service`, самовосстановление и readiness-проверки |

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

## Model Registry: продвижение и откат

Каждое обучение не только логирует run, но и регистрирует новую версию модели
в MLflow Model Registry (`hybrid-review-churn`, `v1`, `v2`, ...). Новая версия
сама по себе ничего не меняет в проде — она просто становится доступной для
сравнения и явного продвижения.

Продвинуть версию в production (или откатиться на более раннюю) — один шаг,
без пересборки образа:

```bash
python scripts/promote_model.py --version 3      # продвинуть v3
python scripts/promote_model.py --version 2      # откатиться на v2
```

Скрипт просто переставляет alias `production` на нужную версию в MLflow — то
же самое можно сделать без кода, в MLflow UI (`Models → hybrid-review-churn →
Add Alias`).

API резолвит модель для инференса не по локальному файлу, а по alias'у
`production` через MLflow Model Registry, и **сам обнаруживает смену версии**:
при каждом запросе (не чаще, чем раз в `MLFLOW_MODEL_REFRESH_SECONDS`, по
умолчанию 30 секунд) сервис лёгким запросом сверяет, на какую версию сейчас
указывает alias, и только если она отличается от закэшированной в памяти —
скачивает новый bundle и подменяет модель. Ручной рестарт или редеплой
сервиса не нужен. Если нужно применить продвижение немедленно, не дожидаясь
истечения интервала:

```bash
curl -X POST http://localhost:8000/admin/reload-model
```

Если `MLFLOW_TRACKING_URI` не задан (например, локальный запуск без Compose),
API прозрачно откатывается на старое поведение — читает
`models/hybrid_churn_bundle.joblib` напрямую с диска.

## Сравнение версии-кандидата с production

Перед тем как продвигать новую версию, её можно оценить на тех же «текущих»
(смещённых) данных, что использует мониторинг дрейфа:

```bash
python scripts/compare_versions.py --candidate 3
# baseline по умолчанию — текущая версия за alias production
python scripts/compare_versions.py --candidate 3 --baseline 2
```

Скрипт скачивает обе версии из Registry по номеру, прогоняет их на
`data/raw/churn_current.csv` и печатает те же метрики (`roc_auc`,
`average_precision`, `sentiment_accuracy`, `sentiment_f1`) бок о бок с
разницей. Так как `churn_current.csv` — синтетические данные с намеренным
сдвигом распределения, сравнение отвечает не просто «какая модель лучше
вообще», а «какая версия увереннее держит удар на данных, непохожих на
обучающие». Это ручной, «по требованию» шаг — естественная точка перед
`promote_model.py`, а не часть автоматического CI.

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

## CI и CD

Workflow `.github/workflows/ci.yml` на каждом pull request и push в `main`:

1. устанавливает зафиксированные зависимости;
2. выполняет `dvc repro`;
3. формирует паспорт релиза;
4. показывает DVC status и метрики;
5. запускает тесты и собирает Docker-образ;
6. прикладывает `dvc.lock`, метрики, metadata, monitoring report и release
   manifest как evidence к запуску GitHub Actions.

Workflow `.github/workflows/cd.yml` запускается автоматически после
**успешного** завершения CI именно на ветке `main` (событие `workflow_run`) —
то есть после мержа, а не на каждый pull request. Он:

1. собирает Docker-образ и публикует его в GHCR
   (`ghcr.io/<repo>:<sha>` и `:latest`);
2. деплоит на сервер по SSH: заходит в `/opt/mlops-student`, обновляет код,
   перезапускает `docker-compose.prod.yml`.

Деплой использует GitHub Environment `production` (можно включить required
reviewers) и секреты `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` —
они привязаны к конкретному репозиторию и не переносятся при форке. В форке
без своих секретов job деплоя просто упадёт на шаге SSH, ничьи чужие сервера
это не затрагивает.

## Kubernetes deploy

После шага Docker Compose тот же `deploy` job в `cd.yml` по SSH раскатывает
`churn-api` в Kubernetes:

1. если на `DEPLOY_HOST` ещё нет `kubectl`/`k3s` — устанавливает k3s
   (однобинарный, однонодовый Kubernetes, официальный скрипт
   `https://get.k3s.io`); повторный запуск безопасен и ничего не переустанавливает;
2. создаёт/обновляет `Secret` `ghcr-pull` (`kubectl create secret
   docker-registry`) из короткоживущего `GITHUB_TOKEN` текущего workflow run —
   нужен, чтобы k3s мог тянуть приватный образ из GHCR;
3. подставляет тег образа, Git SHA и `MLFLOW_TRACKING_URI` в
   `k8s/deployment.yaml` и применяет `k8s/deployment.yaml` + `k8s/service.yaml`;
4. ждёт `kubectl rollout status deployment/churn-api`;
5. делает smoke-test `curl :30080/health` и `curl :30080/predict` прямо на
   хосте (через `Service` типа `NodePort`).

`Deployment` держит **2 реплики** `churn-api` с `readinessProbe`/`livenessProbe`
на `/health` — если реплика падает, Kubernetes перезапускает её сам, а
`Service` продолжает направлять трафик только на готовые Pod'ы. Проверить
руками:

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get pods -l app=churn-api
kubectl describe service churn-api
kubectl delete pod -l app=churn-api --field-selector=status.phase=Running -o name | head -n1 | xargs kubectl delete
kubectl get pods -l app=churn-api -w   # видно, как Deployment поднимает новый Pod
```

MLflow при этом продолжает жить в Docker Compose (`docker-compose.prod.yml`,
порт 5000 на хосте) — в Kubernetes переехал только сам API-сервис, как и в
вебинаре. Pod'ы обращаются к MLflow по IP хоста
(`http://<DEPLOY_HOST>:5000`), а не по DNS-имени контейнера, потому что это
однонодовый k3s без выделенной сети Docker Compose — такое ограничение
типично для учебного/single-node стенда и не подойдёт для мульти-нодового
прод-кластера без отдельного MLflow-сервиса внутри Kubernetes.

CI на каждый pull request дополнительно валидирует манифесты без реального
кластера: `kubectl apply --dry-run=client -f k8s/`.

## Метрики и ограничения

Обучение сохраняет:

- ROC AUC и Average Precision для churn;
- Accuracy и F1 для sentiment;
- размеры train/test выборок.

Данные и отзывы синтетические. Метрики показывают воспроизводимость технического
pipeline, а не готовность модели к реальному бизнес-применению. Для production
понадобятся реальные отзывы, связь с последующим фактом оттока, контроль
дисбаланса, защищённый ingress, аутентификация и постоянное хранилище мониторинга.
