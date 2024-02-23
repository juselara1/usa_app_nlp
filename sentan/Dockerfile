FROM python:3.11.4

RUN mkdir /app
WORKDIR /app/scripts

COPY ./ /app
RUN pip install /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
