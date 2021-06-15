import pandas as pd
import sacrebleu
from statistics import mean
import string
from rouge_metric import PyRouge


#-------Setup---------
MODE = 'top-k'
PATH = 'results/v2/predictions_'+ MODE +'.tsv'
PATH_WMT ='results/v2/predictions_wmt_'+ MODE +'.tsv'

test_df=pd.read_csv('data/test.tsv', sep='\t', encoding='utf8')
predictions_df=pd.read_csv(PATH, sep='\t', index_col=0, encoding='utf8')
predictions_wmt_df=pd.read_csv(PATH_WMT, sep='\t', index_col=0, encoding='utf8')

#Load the WMT 14 newstest data

with open('D:\HAW\Bachelorarbeit\Daten\.sacrebleu\wmt14\de-en.en',encoding='utf8') as f:
    wmt14_en = f.read().splitlines()

#-------Bleu-Score---------

#This needs to be in this Format: [[s1translation1, s2translation1, ...], [s1translation2, s2translation2, ...], ..]
#All alternative transaltion lists need to have the same length 
#Since we have different numbers of translations, some need to be padded
#Padding with the 1st translation won't affect sore see: https://github.com/mjpost/sacreBLEU/issues/48
english_truth=[[],[],[],[],[],[],[],[]]#Max Translations in Test Data = 8
english_preds = []
for i in range(len(predictions_df)):
    y=test_df['eng'].loc[test_df['de']==predictions_df['de'].iloc[i]]
    for k in range(8):
        if len(y)> k:
            english_truth[k].append(y.iloc[k])
        else:
            english_truth[k].append(y.iloc[0])

    english_preds.append(predictions_df[['0','1','2']].iloc[i].to_list())


#WMT 14 Dataset

english_preds_wmt = []

for i in range(len(predictions_wmt_df)):
    english_preds_wmt.append(predictions_wmt_df[['0','1','2']].iloc[i].to_list())


preds_bleu_1 =[]
preds_bleu_3 =[]
preds_bleu_1_wmt =[]
preds_bleu_3_wmt =[]

for x in english_preds:
    preds_bleu_1.append(x[0])
    for i in range(3):
        preds_bleu_3.append(x[i])

for x in english_preds_wmt:
    preds_bleu_1_wmt.append(x[0])
    for i in range(3):
        preds_bleu_3_wmt.append(x[i])

#checking multiple outputs as individual translations against all possible reference translations
#This will multiply the output and reference lists by the number of outputs
english_truth_3 =[]
for i,alt in enumerate(english_truth):
    english_truth_3.append([])
    for x in alt:
        for k in range(3):
            english_truth_3[i].append(x)

wmt14_en_3=[]

for x in wmt14_en:
    for i in range(3):
        wmt14_en_3.append(x)

#Getting the right format for sacrebleu
wmt14_en_3=[wmt14_en_3]
wmt14_en=[wmt14_en]

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

for i in range(len(predictions_df)):
    y_true=test_df['eng'].loc[test_df['de']==predictions_df['de'].iloc[i]].reset_index(drop=True)
    y_pred=predictions_df[['0','1','2']].iloc[i]

    #Remove punctuation, extra whitespaces and set to lowercase
    #See staple paper

    for k, x in enumerate(y_true):
        x=x.translate(str.maketrans('', '', string.punctuation))
        x=x.strip()
        x=x.lower()
        y_true[k]=x
    
    y_true=set(y_true)

    for k, x in enumerate(y_pred):
        x=x.translate(str.maketrans('', '', string.punctuation))
        x=x.strip()
        x=x.lower()
        y_pred[k]=x
    
    y_pred=set(y_pred)   

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


#-------ROUGE---------

references=[]
for i in range(len(predictions_df)):
    for k in range (3):
        references.append(test_df['eng'].loc[test_df['de']==predictions_df['de'].iloc[i]].reset_index(drop=True).to_list())

# Evaluate document-wise ROUGE scores
rouge = PyRouge(rouge_n= False, rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
scores = rouge.evaluate(preds_bleu_3, references)
print(scores)
