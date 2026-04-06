FROM python:3.12-slim
WORKDIR /app
COPY shared/ /app/shared/
COPY services/forex-signals/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/forex-signals/ .
CMD python main.py
