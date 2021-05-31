import pandas as pd
import torch

from transformers import (

    T5TokenizerFast as T5Tokenizer
)

from model import TranslationModel 

import sacrebleu
from statistics import mean


#-------Setup---------

MODE='bs'# Chose decoding Method: bs=Beam Search, dbs=Diverse Beam Search, top-k or top-p Sampling
NUM_OUTPUTS=3
MODEL_NAME= 't5-base'
CKPT_PATH= 'D:/HAW/Bachelorarbeit/Test/results/results/lightning_logs/translation_test/version_0/checkpoints/epoch=2-step=4769.ckpt'

test_df=pd.read_csv('data/test.tsv', sep='\t', index_col=0, encoding='utf8')

#Load the WMT 14 newstest data
with open('D:\HAW\Bachelorarbeit\Daten\.sacrebleu\wmt14\de-en.de',encoding='utf8') as f:
    wmt14_de = f.read().splitlines()

with open('D:\HAW\Bachelorarbeit\Daten\.sacrebleu\wmt14\de-en.en',encoding='utf8') as f:
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

#-------Bleu-Score---------

print('-------Start translating--------')
#This needs to be in this Format: [[s1translation1, s2translation1, ...], [s1translation2, s2translation2, ...], ..]
#All alternative transaltion lists need to have the same length 
#Since we have different numbers of translations, some need to be padded
#Padding with the 1st translation won't affect sore see: https://github.com/mjpost/sacreBLEU/issues/48
english_truth=[[],[],[],[],[],[],[],[]]#Max Translations in Test Data = 8
english_preds = []
for k, x in enumerate(to_english):
    y=test_df['eng'].loc[test_df['de']==x]
    for i in range(8):
        if len(y)> i:
            english_truth[i].append(y.iloc[i])
        else:
            english_truth[i].append(y.iloc[0])

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
#english_truth=english_truth[:2]#Test on first 20 sentences with 2 alternative transaltions
# print(english_preds)
# print(len(english_preds))#should be len(inputs)* num output sentences
# #print(len(english_truth))#should be 8
# print(len(wmt14_en))
preds_bleu_1 =[]
preds_bleu_3 =[]
preds_bleu_1_wmt =[]
preds_bleu_3_wmt =[]

for x in english_preds:
    preds_bleu_1.append(x[0])
    for i in range(NUM_OUTPUTS):
        preds_bleu_3.append(x[i])

for x in english_preds_wmt:
    preds_bleu_1_wmt.append(x[0])
    for i in range(NUM_OUTPUTS):
        preds_bleu_3_wmt.append(x[i])

#checking multiple outputs as individual translations against all possible reference translations
#This will multiply the output and reference lists by the number of outputs
english_truth_3 =[]
for i,alt in enumerate(english_truth):
    english_truth_3.append([])
    for x in alt:
        for k in range(NUM_OUTPUTS):
            english_truth_3[i].append(x)

wmt14_en_3=[]

for x in wmt14_en:
    for i in range(NUM_OUTPUTS):
        wmt14_en_3.append(x)

#Getting the right format for sacrebleu
wmt14_en_3=[wmt14_en_3]
wmt14_en=[wmt14_en]

#print(wmt14_en_3)

de_eng_bleu_1 = sacrebleu.corpus_bleu(preds_bleu_1, english_truth)
de_eng_bleu_3 = sacrebleu.corpus_bleu(preds_bleu_3, english_truth_3)
print("German to English only first translation: ", de_eng_bleu_1.score)
print("German to English all translations: ", de_eng_bleu_3.score)

de_eng_bleu_wmt_1 = sacrebleu.corpus_bleu(preds_bleu_1_wmt, wmt14_en)
de_eng_bleu_wmt_3 = sacrebleu.corpus_bleu(preds_bleu_3_wmt, wmt14_en_3)
print("German to English WMT only first translation: ", de_eng_bleu_wmt_1.score)
print("German to English WMT all translations: ", de_eng_bleu_wmt_3.score)
#-------F1-Score---------
f_scores=[]

for i, x in enumerate(to_english):
    y_true=set(test_df['eng'].loc[test_df['de']==x])
    y_pred=set(english_preds[i])

    tp= len(y_true.intersection(y_pred))
    fp= len(y_pred)-tp
    fn=len(y_true)-tp

    precision=tp / (tp + fp)
    recall=tp / (tp + fn)
    if precision==0 and recall==0:
        f1= 0
        # print(x)
        # print(y_pred)
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    f_scores.append(f1)
    #print(precision, recall, f1)

print('F1-Score:',mean(f_scores))



#-------Saving Predictions---------

df_predictions= pd.DataFrame(english_preds)
df_predictions['de']= to_english
df_predictions.to_csv('results/predictions_'+ MODE +'.tsv' , sep='\t', encoding='utf8')

df_wmt_predictions= pd.DataFrame(english_preds_wmt)
df_wmt_predictions['de']=wmt14_de
df_wmt_predictions.to_csv('results/predictions_wmt_'+ MODE +'.tsv' , sep='\t', encoding='utf8')
