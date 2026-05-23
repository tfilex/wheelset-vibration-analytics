FROM python:3.10-slim

WORKDIR /app
ENV MODEL_CATALOG_MODE=demo
ENV MODEL_CATALOG_LOCKED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src ./src
COPY models/demo_best ./models/demo_best
COPY models/cnn ./models/cnn
COPY models/pred_0 ./models/pred_0
COPY data/raw ./data/raw

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
