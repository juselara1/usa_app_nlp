FROM python:3.11

RUN mkdir /app
WORKDIR /app

COPY ./ /app
# RUN pip install .
RUN apt update -y && apt install make
RUN make install

EXPOSE 8000
# CMD ["streamlit", "run", "scripts/chat.py", "--server.port=8000", "--server.address=0.0.0.0"]
CMD ["make", "run"]
