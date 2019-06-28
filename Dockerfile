FROM python:2
ADD api.py /
RUN pip install -r requirements.txt
CMD [ "python", "./api.py" ]