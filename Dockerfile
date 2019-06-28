FROM python:2.7.16-alpine3.9
ADD api.py /
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
CMD [ "python", "api.py" ]
