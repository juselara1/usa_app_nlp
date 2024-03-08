FROM python:3.11

RUN mkdir /app
WORKDIR /app

COPY ./ /app

RUN apt update -y && apt install make
RUN make install

EXPOSE 8000
CMD ["make", "run"]
