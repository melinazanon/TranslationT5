import pandas as pd
import torch

from transformers import (

    T5TokenizerFast as T5Tokenizer
)

from model import TranslationModel 

import argparse

#-------Setup---------
parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='bs')
args = parser.parse_args()
print('Generation mode: '+ args.mode)
MODE=args.mode# Chose decoding Method: bs=Beam Search, dbs=Diverse Beam Search, top-k or top-p Sampling
NUM_OUTPUTS=3
MODEL_NAME= 't5-base'
CKPT_PATH= 'D:/HAW/Bachelorarbeit/Test/results/results/lightning_logs/translation_test/version_0/checkpoints/epoch=2-step=4769.ckpt'

test_df=pd.read_csv('data/test.tsv', sep='\t', index_col=0, encoding='utf8')

#Load the WMT 14 newstest data
with open('wmt14\de-en.de',encoding='utf8') as f:
    wmt14_de = f.read().splitlines()

with open('wmt14\de-en.en',encoding='utf8') as f:
    wmt14_en = f.read().splitlines()


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

trained_model = TranslationModel.load_from_checkpoint(CKPT_PATH)

torch.manual_seed(0)
#-------Generation/Decoding---------

def translate(text, mode):
    do_sample=False
    top_k=0
    top_p=1
    num_beams=1
    num_beam_groups=1
    diversity_penalty=0
    early_stopping=False

    if mode =='bs':
        num_beams=10
        early_stopping=True
    if mode == 'dbs':
        num_beams=10
        early_stopping=True
        num_beam_groups=10
        diversity_penalty=1.5
    if mode == 'top-k':
        do_sample=True
        top_k= 10
    if mode == 'top-p':
        do_sample=True
        top_p=0.9

    text_encoding = tokenizer(
        text,
        max_length=180,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = trained_model.model.generate(
        input_ids = text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        do_sample= do_sample,
        top_k= top_k,
        top_p= top_p,
        num_beams= num_beams,
        num_beam_groups= num_beam_groups,
        diversity_penalty=diversity_penalty,
        num_return_sequences=NUM_OUTPUTS, #Multiple sentence output
        repetition_penalty=1.5,
        early_stopping=early_stopping
    )
    
    #Decode token ids
    preds=[]
    for gen_id in generated_ids:
        preds.append(tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    
    return preds

#Inputs
to_english = test_df['de'].unique().tolist()
#to_english = to_english[20:40]

#-------Example Translation---------
sample_row = test_df.iloc[49]
text= sample_row["de"]
print('Example Sentence:', text)

print('Translation:',translate(text,MODE))

#-------Translating---------

print('-------Start translating--------')

english_preds = []
for k, x in enumerate(to_english):
    print('translating'+ str(k)+'/'+ str(len(to_english)))
    english_preds.append(translate(x, MODE))
    

#WMT 14 Dataset
# wmt14_de= wmt14_de[:20]
# wmt14_en = wmt14_en[:20]
english_preds_wmt = []
# print(wmt14_en)
print('-------Start translating WMT--------')

for k, x in enumerate(wmt14_de):
    print('translating'+ str(k)+'/'+ str(len(wmt14_de)))
    english_preds_wmt.append(translate(x, MODE))

print('-------Finished translating--------')

#-------Saving Predictions---------

df_predictions= pd.DataFrame(english_preds)
df_predictions['de']= to_english
df_predictions.to_csv('results/v2/predictions_'+ MODE +'.tsv' , sep='\t', encoding='utf8')

df_wmt_predictions= pd.DataFrame(english_preds_wmt)
df_wmt_predictions['de']=wmt14_de
df_wmt_predictions.to_csv('results/v2/predictions_wmt_'+ MODE +'.tsv' , sep='\t', encoding='utf8')
