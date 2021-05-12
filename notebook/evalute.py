import pandas as pd

from transformers import (

    T5TokenizerFast as T5Tokenizer
)

from model import TranslationModel 

import sacrebleu

test_df=pd.read_csv('data/test.tsv', sep='\t', index_col=0, encoding='utf8')
MODEL_NAME= 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

#TODO: Change this to load from Chekpoint after Training
model= TranslationModel()

#-------Generation/Decoding---------
#TODO: find good generation parameters (num outputs, beam or no beam, length and repetion penalty etc.)

def translate(text):
    text_encoding = tokenizer(
        text,
        max_length=512,#This is way to much, max length in data = 110
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = model.model.generate(
        input_ids = text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=5,
        num_return_sequences=1, #Multiple sentence output
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    #TODO: Change this part to not be one long String
    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]

    return " ".join(preds)

#-------Example Translation---------
# sample_row = test_df.iloc[67]
# text= sample_row["de"]
# print(text)

# print(translate(text))

#-------Bleu-Score---------
#TODO: check multiple outputs as individual translations against all possible reference translations
#This will multiply the output and reference lists by the number of outputs

#Inputs
to_english = test_df['de'].unique().tolist()
to_english = to_english[:20]

#This needs to be in this Format: [[s1translation1, s2translation1, ...], [s1translation2, s2translation2, ...], ..]
#All alternative transaltion lists need to have the same length 
#Since we have different numbers of translations, some need to be padded
#Padding with the 1st translation won't affect sore see: https://github.com/mjpost/sacreBLEU/issues/48
english_truth=[[],[],[],[],[],[],[],[]]#Max Translations in Test Data = 8
english_preds = []
for x in to_english:
    y=test_df['eng'].loc[test_df['de']==x]
    for i in range(8):
        if len(y)> i:
            english_truth[i].append(y.iloc[i])
        else:
            english_truth[i].append(y.iloc[0])
    english_preds.append(translate(x))

english_truth=english_truth[:2]#Test on first 20 sentences with 2 alternative transaltions
print(english_preds)
print(len(english_preds))#should be len(inputs)* num output sentences
print(len(english_truth))#should be 8
de_eng_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
print("German to English: ", de_eng_bleu.score)

