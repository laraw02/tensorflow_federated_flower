FROM python:3

WORKDIR /usr/src/app

ENV client_number = 1

COPY requirements_client.txt ./
RUN apt-get update && apt-get install libgl1 libhdf5-dev -y
RUN pip install -r requirements_client.txt


COPY . ./

CMD ["python", "client.py"]
