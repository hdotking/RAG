.PHONY: up down build run

# Start Docker Compose - detach mode
up:
	docker-compose up -d

# Stop Docker Compose
down:
	docker-compose down

# Run the script
run:
	poetry run python vector_db.py

# Build Docker images
build:
	docker-compose build

