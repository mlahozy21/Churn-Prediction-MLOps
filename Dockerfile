FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
# Train a model at build time so the image is self-contained (uses synthetic data
# if no real CSV is mounted). Override by mounting a trained model at /app/models.
RUN python -m churn.train --no-mlflow --n-synthetic 5000
EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
