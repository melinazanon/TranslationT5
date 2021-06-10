FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
 

COPY requirements.txt ./

COPY model.py ./
COPY train.py ./

COPY data ./data

RUN pip install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

CMD ["python3", "train.py"]