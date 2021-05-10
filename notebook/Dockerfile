FROM pytorch/pytorch:latest

COPY requirements.txt ./

COPY model.py ./
COPY train.py ./

COPY data ./data

RUN pip install -r requirements.txt

CMD ["python3", "train.py"]