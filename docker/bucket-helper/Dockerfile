FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/bucket-helper

COPY requirements.txt /opt/bucket-helper/requirements.txt

RUN pip install --no-cache-dir -r /opt/bucket-helper/requirements.txt

ENTRYPOINT ["python"]
