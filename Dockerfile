FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and pre-trained model artifacts
COPY app.py .
COPY lda_model.pkl .
COPY vectorizer.pkl .
COPY topic_labels.json .

EXPOSE 5000

# Gunicorn serves the app in production (Flask's built-in server is dev-only)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
