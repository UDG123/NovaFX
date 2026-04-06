FROM python:3.12-slim
WORKDIR /app
COPY shared/ /app/shared/
COPY services/telegram-bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/telegram-bot/ .
CMD python main.py
