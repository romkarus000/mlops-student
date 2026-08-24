.PHONY: setup data train monitor test serve up

setup:
	python -m pip install -r requirements.txt

data:
	python scripts/generate_data.py

train:
	python -m src.churn_ml.training

monitor:
	python -m src.churn_ml.monitoring

test:
	pytest -q

serve:
	uvicorn src.churn_ml.api:app --reload --port 8000

up:
	docker compose up --build
