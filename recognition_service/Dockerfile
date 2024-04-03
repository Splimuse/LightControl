FROM python:3-slim-buster

RUN mkdir /code

WORKDIR /code

RUN apt-get update -y && \
    apt-get install -y gcc && \
    apt-get install -y libffi-dev

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8081"]
