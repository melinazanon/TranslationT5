import pandas as pd
import torch
from torch.utils.data import Dataset , DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)


MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class TranslationDataset(Dataset):
    
    def __init__(
        self,
        data:pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int=110,
        translation_max_token_len: int=80,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.translation_max_token_len = translation_max_token_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        #Read line of DataFrame
        data_row= self.data.iloc[index]

        text= data_row['de']

        #Tokenize sentences
        #Pad to Max length
        #Return pytorch tensors
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
            data_row["eng"],
            max_length = self.translation_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        #Set <pad> Tokens (0) in Translation to -100
        #This will be ignored as input
        labels = translation_encoding["input_ids"]
        labels[labels ==0]= -100

        return dict(
            text=text,
            translation=data_row["eng"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=translation_encoding["attention_mask"].flatten()
        )

class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size = 8,
        text_max_token_len: int=110,
        translation_max_token_len: int=80
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.translation_max_token_len = translation_max_token_len
        
    def setup(self, stage= None):
        self.train_dataset = TranslationDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.translation_max_token_len,
        )
        
        self.test_dataset = TranslationDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.translation_max_token_len,
        )

        self.val_dataset = TranslationDataset(
            self.val_df,
            self.tokenizer,
            self.text_max_token_len,
            self.translation_max_token_len,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=18
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=18
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=18
        )


class TranslationModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        #Define model structure from pretrained T5
        self.lr=0.0003
    
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        
        output= self.model(
            input_ids,
            attention_mask= attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask= batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask = attention_mask, 
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask= batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask = attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )

        self.log("val_loss", loss, prog_bar=True, logger=True,sync_dist=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) :
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask= batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask = attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )

        self.log("test_loss", loss, prog_bar=True, logger=True,sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

