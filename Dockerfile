FROM python:2.7.16-alpine3.9
ADD api.py /
RUN pip install -r requirements.txt
CMD [ "python", "./api.py" ]
