FROM python:3.11.4

RUN mkdir /app
WORKDIR /app
COPY ./ /app
RUN make install
RUN make install-playwright
RUN rm -rf /app/*
