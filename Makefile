.PHONY: install test lint format docker-up docker-down

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run flake8 src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
