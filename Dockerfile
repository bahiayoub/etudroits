FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# PyTorch en version CPU uniquement (beaucoup plus léger)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Interface web, dans une couche séparée : l'ajouter ne réinstalle pas tout le reste
COPY requirements-app.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements-app.txt

COPY . .

CMD ["bash"]
