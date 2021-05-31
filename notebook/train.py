import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger 

print('cuda:'  + str(torch.cuda.is_available()))

from transformers import (

    T5TokenizerFast as T5Tokenizer
)

from model import TranslationModel , TranslationDataModule

train_df=pd.read_csv('data/train.tsv', sep='\t', index_col=0, encoding='utf8')
test_df=pd.read_csv('data/test.tsv', sep='\t', index_col=0, encoding='utf8')
val_df=pd.read_csv('data/val.tsv', sep='\t', index_col=0, encoding='utf8')

N_GPUS=-1
N_EPOCHS= 20
BATCH_SIZE =8
MODEL_NAME= 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

if __name__ =='__main__':
    pl.seed_everything(42)
    data_module = TranslationDataModule(train_df, test_df, val_df, tokenizer,batch_size=BATCH_SIZE)

    model= TranslationModel()

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )

    logger= TensorBoardLogger("results/lightning_logs", name="translation_test")

    trainer = pl.Trainer(
        default_root_dir='results/checkpoints',
        logger=logger,
        callbacks=[early_stop_callback],
        max_epochs=N_EPOCHS,
        gpus=N_GPUS,
        #auto_lr_find=True,  
        #accelerator='ddp',
        progress_bar_refresh_rate=10
      
    )

    trainer.fit(model, data_module)
    #trainer.save_checkpoint("test1.ckpt")
    #trainer.tune(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    