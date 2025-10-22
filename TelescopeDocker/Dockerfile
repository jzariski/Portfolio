FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install -r requirements.txt
COPY your_code/ ./your_code/
COPY api/ ./api/
RUN mkdir -p models
EXPOSE 8000
WORKDIR /app/api
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app"]