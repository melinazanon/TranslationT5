import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 


from transformers import (

    T5TokenizerFast as T5Tokenizer
)

from model import TranslationModel , TranslationDataModule

train_df=pd.read_csv('data/train.tsv', sep='\t', index_col=0, encoding='utf8')
test_df=pd.read_csv('data/test.tsv', sep='\t', index_col=0, encoding='utf8')
val_df=pd.read_csv('data/val.tsv', sep='\t', index_col=0, encoding='utf8')

N_EPOCHS= 1
BATCH_SIZE =8
MODEL_NAME= 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
data_module = TranslationDataModule(train_df, test_df, val_df, tokenizer,batch_size=BATCH_SIZE)

model= TranslationModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="/checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger= TensorBoardLogger("lightning_logs", name="translation_test")

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)