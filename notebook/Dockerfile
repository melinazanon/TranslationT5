FROM pytorchlightning/pytorch_lightning:base-cuda-py3.7-torch1.8
 

COPY requirements.txt ./

COPY model.py ./
COPY train.py ./

COPY data ./data

RUN pip install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

CMD ["python3", "train.py"]