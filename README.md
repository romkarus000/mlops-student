# ml-service

Учебный MLOps-стенд: сервис кредитного скоринга, собранный вручную инструмент за
инструментом — от контейнера до CI/CD с деплоем. Использовался как рабочий проект
для вебинара по MLOps; этот README — методичка, свёрнутая до состояния документации
репозитория: что тут есть, как это поднять с нуля и на какие грабли реально
наступали при сборке.

```
Docker → FastAPI → DVC → MLflow → Evidently → GitHub Actions (CI/CD)
```

Ключевая идея всего стенда: это не пять изолированных демо-стендов, а один
наращиваемый сервис. Контейнер, поднятый в разделе Docker, остаётся жить до конца —
FastAPI, DVC, MLflow и Evidently достраиваются вокруг него, а не поднимают
параллельные копии.

## Содержание

- [Архитектура](#архитектура)
- [Быстрый старт с нуля](#быстрый-старт-с-нуля)
- [Docker](#docker)
- [FastAPI-сервис](#fastapi-сервис)
- [DVC — данные и пайплайн обучения](#dvc--данные-и-пайплайн-обучения)
- [MLflow — трекинг и реестр моделей](#mlflow--трекинг-и-реестр-моделей)
- [Evidently — мониторинг дрифта](#evidently--мониторинг-дрифта)
- [CI/CD — GitHub Actions](#cicd--github-actions)
- [Паспорт релиза](#паспорт-релиза)
- [Kubernetes (опционально)](#kubernetes-опционально)
- [Известные ограничения и нюансы](#известные-ограничения-и-нюансы)

## Архитектура

| Файл/директория | Роль |
|---|---|
| `Dockerfile`, `app/main.py` | FastAPI-сервис — точка входа для модели |
| `requirements.txt` | Зависимости **сервиса** (то, что едет в образ) |
| `data/train.csv` | Обучающий датасет, версионируется через DVC |
| `dvc.yaml`, `params.yaml`, `dvc.lock` | Пайплайн обучения: структура, гиперпараметры, зафиксированные хеши прогона |
| `src/train.py` | Код обучения модели + логирование в MLflow |
| `monitor.py` | Evidently-отчёт и CI-гейт по дрифту данных |
| `scripts/gen_train_data.py` | Детерминированный генератор `data/train.csv` — используется в CI вместо сетевого `dvc pull` |
| `scripts/gen_batch.py` | Детерминированный генератор "продового" батча для дрифт-гейта в CI |
| `scripts/build_release_manifest.py` | Паспорт релиза: коммит, DVC-состояние, модель из Registry, метрики, образ |
| `scripts/register_model.py` | Регистрация обученной модели в MLflow Model Registry + алиас `production` |
| `.github/workflows/ci.yml` | CI (сборка, тесты, дрифт-гейт) + CD (деплой на прод-сервер) |

## Быстрый старт с нуля

Стенд рассчитан на чистую VM с Ubuntu 24.04. Все команды — от `root`.

```bash
git clone <URL-этого-репозитория> ml-service
cd ml-service
```

Дальше по порядку — разделы ниже; каждый содержит установку, настройку и способ
проверить, что шаг реально сработал.

## Docker

**Установка Docker Engine** (официальный репозиторий, не `snap` и не пакет дистрибутива):

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  apt-get remove -y $pkg
done

apt-get -o DPkg::Lock::Timeout=300 update
apt-get -o DPkg::Lock::Timeout=300 install -y ca-certificates curl

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get -o DPkg::Lock::Timeout=300 update
apt-get -o DPkg::Lock::Timeout=300 install -y \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

docker run --rm hello-world
```

> **Нюанс.** На свежих облачных образах в первые минуты после старта работает
> `unattended-upgrades` и держит `dpkg`-лок — без флага `-o DPkg::Lock::Timeout=300`
> `apt-get install` сразу падает с `Could not get lock`. Если после установки
> `systemctl status docker` показывает `failed (service-start-limit-hit)` —
> `systemctl reset-failed docker.socket docker.service && systemctl restart docker`.

**Сборка и запуск:**

```bash
docker pull python:3.11-slim   # если 429 Too Many Requests — см. раздел "Известные ограничения"
docker build -t ml-service:0.1 .
docker run -d -p 8000:8000 --name ml ml-service:0.1

curl localhost:8000/health
```

## FastAPI-сервис

Два эндпоинта:

- `GET /health` — проверка живости, её же читает readiness-проба Kubernetes.
- `POST /predict?alias=production` — предсказание модели. Параметр `alias`
  (`production` по умолчанию, либо `staging`) определяет, какую версию модели из
  MLflow Model Registry загружать — переключение версий без пересборки образа.

```bash
curl -X POST "localhost:8000/predict?alias=production" \
  -H "Content-Type: application/json" \
  -d '{"features":[35, 60000, 10, 15000]}'
# {"prediction": 0.0, "model_alias": "production"}
```

Признаки на входе, в порядке: `age`, `income`, `credit_history_years`, `loan_amount`.

Автосгенерированная документация — `/docs` (Swagger UI), строится из Pydantic-схемы
`PredictRequest` без единой дополнительной строки кода.

Модель кэшируется в памяти процесса на алиас — при первом запросе с новым алиасом
она грузится из MLflow, дальше переиспользуется. Если алиас в Registry подвинули на
другую версию — уже запущенный сервис продолжит отдавать старую до перезапуска
контейнера (см. «Известные ограничения»).

## DVC — данные и пайплайн обучения

```bash
apt-get -o DPkg::Lock::Timeout=300 install -y python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install dvc pandas numpy scikit-learn pyyaml

mkdir -p ~/dvc-storage   # локальный remote для учебного стенда
dvc pull   # или python3 scripts/gen_train_data.py, если remote недоступен
dvc repro  # обучает модель по params.yaml, пишет models/model.pkl и metrics.json
```

`params.yaml` — гиперпараметры (`n_estimators`, `max_depth`), отдельно от кода
`src/train.py`: DVC отслеживает изменения этого файла как зависимость стадии
`train` и пересчитывает пайплайн только когда параметры реально изменились.
`dvc.lock` фиксирует, какие именно хеши данных/кода/параметров дали текущий
результат — это и есть журнал экспериментов на уровне файлов
(`dvc metrics diff` сравнивает прогоны).

## MLflow — трекинг и реестр моделей

```bash
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &

python3 src/train.py     # логирует run в эксперимент "credit-scoring"
python3 scripts/register_model.py   # регистрирует модель, ставит алиас production
```

UI — `http://localhost:5000`. Начиная с MLflow 3.x сервер защищён от DNS rebinding
и принимает только `Host: localhost`/`127.0.0.1` — открыть его по внешнему IP
напрямую не выйдет (`403`); для доступа снаружи используйте SSH-туннель:

```bash
ssh -L 5000:localhost:5000 root@<ip-сервера>
# и открывайте http://localhost:5000 у себя
```

Версии модели адресуются через алиасы, а не через "stage" (устаревший API):

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("credit-model", "production", 2)  # перевод прода на версию 2
client.set_registered_model_alias("credit-model", "staging", 1)     # откат/кандидат
```

## Evidently — мониторинг дрифта

```bash
python3 scripts/gen_batch.py   # "продовый" батч (в CI — без искусственного сдвига)
python3 monitor.py             # отчёт + гейт с кодом возврата для CI
```

`monitor.py` строит `DataDriftPreset` (K-S тест по числовым признакам), сохраняет
`drift_report.html` и завершается с ненулевым кодом, если доля задрифтивших колонок
превышает порог — именно этот код возврата CI использует, чтобы блокировать деплой
до, а не после того, как дрифт попал в прод.

## CI/CD — GitHub Actions

`.github/workflows/ci.yml`, две джобы:

**`build-and-test`** (push/PR в `main`): устанавливает зависимости → обучает модель
на детерминированных данных (`scripts/gen_train_data.py`, без сетевого `dvc pull` —
см. нюансы) → регистрирует её в MLflow с алиасом `production` → прогоняет
Evidently-гейт → собирает Docker-образ → smoke-тестит контейнер настоящим вызовом
`/predict?alias=production` → пушит образ в GHCR (`ghcr.io/<repo>:latest`, только на
`main`).

**`deploy`** (`needs: build-and-test`, только `main`): по SSH разворачивает свежий
образ из GHCR на целевом сервере. **Выключен по умолчанию** — пока деплоить некуда,
джоба пропускается (`skipped`), и прогон остаётся зелёным. Когда появится сервер:

```bash
gh secret set DEPLOY_HOST -b"<ip-прод-сервера>"
gh secret set DEPLOY_SSH_KEY < ~/.ssh/deploy_key
gh secret set DEPLOY_USER -b"<пользователь>"   # необязательно, по умолчанию root
gh variable set DEPLOY_ENABLED -b true
```

после чего джоба подключится, остановит старый контейнер и поднимет новый.

## Паспорт релиза

Один JSON-файл, который связывает коммит, состояние DVC-пайплайна, обученную
модель, метрики и тег образа — чтобы про любую поставку можно было доказать, из чего
она собрана.

```bash
python scripts/build_release_manifest.py                 # -> reports/release_manifest.json
python scripts/build_release_manifest.py --skip-registry # без запроса к MLflow Registry
```

Перед запуском должен быть выполнен `dvc repro` (нужны `models/model.pkl`,
`metrics.json`, `data/train.csv`). В паспорт попадают: SHA коммита и версия сервиса
(`SERVICE_VERSION`, по умолчанию `0.1.0`); SHA256 `dvc.lock`, `params.yaml`,
`requirements.txt`; SHA256 `data/train.csv`; SHA256 `models/model.pkl` и версия/`run_id`
модели `credit-model@production` из MLflow Registry; метрики из `metrics.json`; тег
образа (`IMAGE_TAG`). Недостающий входной файл — ошибка с подсказкой, а не пустое поле.

В CI паспорт строится после smoke-теста и прикладывается к запуску артефактом
`release-manifest`.

## Kubernetes (опционально)

Локальный кластер через `minikube` (учебный, не production):

```bash
minikube start --driver=docker --force
minikube image load ml-service:0.1   # свой Docker-демон у minikube — образ нужно занести явно
kubectl apply -f deployment.yaml     # Deployment (2 реплики, readinessProbe на /health) + Service
```

## Известные ограничения и нюансы

- **`docker pull python:3.11-slim` → `429 Too Many Requests`.** Лимит анонимных
  pull'ов Docker Hub. Обход: `docker pull mirror.gcr.io/library/python:3.11-slim &&
  docker tag mirror.gcr.io/library/python:3.11-slim python:3.11-slim`.
- **MLflow-контейнер и модель по алиасу: `--network host`, а не проброс портов.**
  MLflow 3.x проверяет заголовок `Host` у входящих запросов и пропускает только
  `localhost`; `host.docker.internal` эту проверку не проходит. Контейнер сервиса
  нужно поднимать с `--network host`.
- **`No such artifact: ''` при загрузке модели.** MLflow с локальным (файловым)
  artifact store хранит путь к артефактам как путь на диске — клиенту нужен
  реальный доступ к этой файловой системе, не только к API сервера. Решение —
  монтировать `mlruns` в контейнер по тому же абсолютному пути:
  `-v /root/ml-service/mlruns:/root/ml-service/mlruns`. С реальным облачным remote
  (S3/GCS) этой проблемы нет.
- **DVC-remote по SSH между разными облаками — ненадёжен.** Между GitHub Actions
  (Azure) и одиночной VM ловили то таймауты, то обрыв соединения после успешного
  логина (похоже на MTU на стыке облаков). В CI это обойдено генерацией данных на
  месте (`scripts/gen_train_data.py`); шаг с SSH-remote оставлен в workflow как
  рабочая демонстрация механики, помечен `continue-on-error: true`.
- **`git init` без `-b main` создаёт `master`.** CI и `gh pr create` в этом
  репозитории написаны под ветку `main`.
- **GHCR-пуш: `denied: installation not allowed to Create organization package`.**
  Нужен явный `permissions: packages: write` у джобы в workflow — без него
  автоматический `GITHUB_TOKEN` может не иметь права записи в GitHub Packages.

---

Проект собирался как учебный материал; часть решений (локальный DVC-remote,
одиночная VM вместо managed-инфраструктуры) — упрощения ради самодостаточности
демонстрации, а не production-практика.
