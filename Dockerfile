FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 5000

WORKDIR $APP_HOME

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 app:app