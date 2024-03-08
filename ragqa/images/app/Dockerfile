FROM python:3.11

RUN mkdir /app
WORKDIR /app

COPY .streamlit/ /root/.streamlit/
COPY . /app
RUN make install

EXPOSE 8000
CMD ["make", "run"]
