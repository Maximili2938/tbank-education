# docker/Dockerfile
FROM python:3.8-slim

# Установка зависимостей системного уровня
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY ../requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY .. .

# Экспонирование порта
EXPOSE 5000

# Команда для запуска Flask API
CMD ["python", "app.py"]
