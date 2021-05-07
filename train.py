import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset , DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

from tqdm.auto import tqdm

tokenizer= T5Tokenizer

class TranslationDataset(Dataset):
    
    def __init__(
        self,
        data:pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int=128,
        translation_max_token_len: int=128,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.translation_max_token_len = translation_max_token_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row= self.data.iloc[index]

        text= data_row['eng']

        text_encoding = tokenizer(
            text,
            max_length = self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        translation_encoding = tokenizer(
            data_row["de"],
            max_length = self.translation_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = translation_encoding["input_ids"]
        labels[labels ==0]= -100

        return dict(
            text=text,
            translation=data_row["de"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=translation_encoding["attention_mask"].flatten()
        )