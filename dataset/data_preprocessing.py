'''
Author: sicen Liu
Date: 2021-08-28 10:35:39
LastEditTime: 2021-08-28 10:37:48
'''
import numpy as np
import pandas as pd
import os
import json
import csv
from tqdm import tqdm
import dill
from random import shuffle
import random
import pickle
from transformers import BertTokenizer, BertModel
import nltk
tokenizer = BertTokenizer.from_pretrained("download the BERT model, we used the /bert-base-uncased/")
nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')
MIMIC_DATA_PATH = "MIMIC_III data dir"

med_file = os.path.join(MIMIC_DATA_PATH, 'PRESCRIPTIONS.csv')
diag_file = os.path.join(MIMIC_DATA_PATH, 'DIAGNOSES_ICD.csv')
proc_file = os.path.join(MIMIC_DATA_PATH, 'PROCEDURES_ICD.csv')
labtest_file = os.path.join(MIMIC_DATA_PATH, 'LABEVENTS.csv')
notes_file = os.path.join(MIMIC_DATA_PATH, 'NOTEEVENTS.csv')
adm_file = os.path.join(MIMIC_DATA_PATH, 'ADMISSIONS.csv')
ndc2atc_file = "./ndc2atc_level4.csv"
cid_atc = './drug-atc.csv'
ndc2rxnorm_file = './ndc2rxnorm_mapping.txt'

admission_data = pd.read_csv(os.path.join(MIMIC_DATA_PATH, "ADMISSIONS.csv"))
patients = pd.read_csv(os.path.join(MIMIC_DATA_PATH, "PATIENTS.csv"))
list(admission_data)
t = admission_data.SUBJECT_ID.value_counts()
count_single = []
count_multi = []
single_patient_ids = []
multi_patient_ids = []
for k, v in t.items():
    if v > 1:
        multi_patient_ids.append(k)
        count_multi.append(v)
    else:
        single_patient_ids.append(k)
        count_single.append(v)
print("total patient number", len(patients),"\nsingle visit patient number:",len(count_single), "\nmulti visit patient number",len(count_multi))
print("admission patient number is:", len(admission_data.SUBJECT_ID.unique()))
print(np.mean(count_single), np.mean(count_multi))


def process_diag(diag_file):
    print('process_diag')

    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    diag_pd.drop(columns=['SEQ_NUM'], inplace=True)
    return diag_pd.reset_index(drop=True)

def process_procedure(procedure_file):
    print("process procedure")
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    return pro_pd

def ndc2atc4(med_pd, ndc_rxnorm_file):
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

def process_labtest(labtest_file):
    print("process labtest")
    labtest_pd = pd.read_csv(labtest_file)
    labtest_pd.drop(columns=['ROW_ID', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG'], axis=1, inplace=True)
    labtest_pd.fillna(method='pad', inplace=True)
    labtest_pd.dropna(inplace=True)
    labtest_pd['CHARTTIME'] = pd.to_datetime(
        labtest_pd['CHARTTIME'], format='%Y-%m-%d %H:%M:%S')
    labtest_pd.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTTIME'], inplace=True)
    labtest_pd = labtest_pd.groupby(by=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'])['ITEMID'].unique().reset_index()
    labtest_pd['ITEMID'] = labtest_pd['ITEMID'].map(lambda x: list(x))
    labtest_pd.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTTIME'], inplace=True)
    labtest_pd.drop(columns=['CHARTTIME'], inplace=True)
    labtest_pd = labtest_pd.groupby(by=['SUBJECT_ID','HADM_ID']).apply(lambda x : x['ITEMID'].tolist()).reset_index()
    labtest_pd.columns = ['SUBJECT_ID','HADM_ID','ITEMID']
    labtest_pd['itemLen'] = labtest_pd['ITEMID'].map(lambda x: len(x))
    return labtest_pd    

def process_med(med_file):
    # 处理medication数据
    print('prcess med...')
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                            'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                            'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                            'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(
        med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID',
                            'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    med_pd.drop(columns=['ICUSTAY_ID'], inplace=True)
    med_pd = ndc2atc4(med_pd, ndc2rxnorm_file)
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID', 'STARTDATE'])['NDC'].unique().reset_index()
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'STARTDATE'], inplace=True)
    med_pd.drop(columns=['STARTDATE'], inplace=True)
    med_pd = med_pd[['SUBJECT_ID', 'HADM_ID', 'NDC']]
    new_med_pd = med_pd.groupby(by=['SUBJECT_ID','HADM_ID']).apply(lambda x : x['NDC'].tolist()).reset_index()
    new_med_pd.columns = ['SUBJECT_ID', 'HADM_ID', 'NDC']
    new_med_pd['NDC_datLen'] = new_med_pd['NDC'].map(lambda x: len(x))
    return new_med_pd

def tokenizer_text(note):
    generator = nlp_tool.span_tokenize(note)
    all_sents_inds = []
    for t in generator:
        all_sents_inds.append(t)

    text = ""
    for ind in range(len(all_sents_inds)):
        start = all_sents_inds[ind][0]
        end = all_sents_inds[ind][1]

        sentence_txt = note[start:end]

        tokens = [t.lower() for t in tokenizer.tokenize(sentence_txt) if not t.isnumeric()]
        if ind == 0:
            text += '[CLS] ' + ' '.join(tokens) + ' [SEP]'
        else:
            text += ' [CLS] ' + ' '.join(tokens) + ' [SEP]'

    text = '"' + text + '"'
    return text

def process_notes(notes_file):
    print("process notes")
    notes = pd.read_csv(notes_file)
    notes.drop(columns=['ROW_ID','CHARTTIME','STORETIME','DESCRIPTION','CGID','ISERROR'], inplace=True)
    notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'], inplace=True)
    notes.drop(columns=['CHARTDATE'], inplace=True)
    # notes['TEXT'] = notes['TEXT'].apply(lambda x : tokenizer_text(x)) # 如果想要一开始就将文本进行分句并tokenized 取消这一句的注释
    notes['CATEGORY_TEXT'] = notes.apply(lambda x: (x['CATEGORY'], x['TEXT']), axis=1)
    notes.drop(columns=['CATEGORY','TEXT'], inplace=True)
    notes = notes.groupby(by=['SUBJECT_ID','HADM_ID']).apply(lambda x: x['CATEGORY_TEXT'].tolist()).reset_index()
    notes.columns = ['SUBJECT_ID','HADM_ID','CATEGORY_TEXT']
    notes['TEXT_len'] = notes['CATEGORY_TEXT'].map(lambda x: len(x))
    return notes
def process_adm(adm_file):
    print('process admission')
    df_adm = pd.read_csv(adm_file)
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    # print(df_adm)

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
    # df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

    # rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    # df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    # df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    #When we filter out the "ELECTIVE", we need to correct the next admit time for these admissions since there might be 'emergency' next admit after "ELECTIVE"
    # df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
    df_adm['READMISSION_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
    ### filter out newborn and death
    # df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
    # df_adm = df_adm[df_adm.DEATHTIME.isnull()]
    df_adm['DURATION'] = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)
    return df_adm[['SUBJECT_ID','HADM_ID','READMISSION_LABEL','HOSPITAL_EXPIRE_FLAG', 'DURATION']]


adm_pd = process_adm(adm_file)
diag_pd = process_diag(diag_file)
proc_pd = process_procedure(proc_file)
labtest_pd = process_labtest(labtest_file)
med_pd = process_med(med_file)
notes_pd = process_notes(notes_file)

adm_pd_key = adm_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
proc_pd_key = proc_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
labtest_pd_key = labtest_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
notes_key = notes_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
combined_key = combined_key.merge(proc_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
combined_key = combined_key.merge(labtest_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
combined_key = combined_key.merge(notes_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
print("combined_key len:",len(combined_key))

diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
proc_pd = proc_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
labtest_pd = labtest_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
notes_pd = notes_pd.merge(combined_key, on=['SUBJECT_ID','HADM_ID'], how='inner')
diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'DIAG_CODE'})
diag_pd['DIAG_CODE'] = diag_pd['DIAG_CODE'].map(lambda x: list(x))
proc_pd = proc_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})
proc_pd['PRO_CODE'] = proc_pd['PRO_CODE'].map(lambda x: list(x))
print(list(med_pd), list(proc_pd), list(labtest_pd), list(notes_pd))
len(med_pd), len(notes_pd), len(diag_pd), len(proc_pd), len(labtest_pd)

def write_ids(data, save_path):
    with open(save_path, 'w') as f:
        for d in data:
            f.write('{}\n'.format(d))

# adm_pd = process_adm(adm_file)
adm_pd = adm_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
adm_pd
adm_subj = adm_pd.SUBJECT_ID.value_counts()
# adm_subj
count_single = []
count_multi = []
single_patient_ids = []
multi_patient_ids = []
for k, v in adm_subj.items():
    if v > 1:
        multi_patient_ids.append(k)
        count_multi.append(v)
    else:
        single_patient_ids.append(k)
        count_single.append(v)
print("total patient number", len(patients),"\nsingle visit patient number:",len(count_single), "\nmulti visit patient number",len(count_multi))
print("selected admission patient number is:", len(adm_pd.SUBJECT_ID.unique()))
print(np.mean(count_single), np.mean(count_multi))
write_ids(multi_patient_ids, './multi_patient_ids.txt')
write_ids(single_patient_ids, './single_patient_ids.txt')

merge_pd = diag_pd.merge(med_pd, on=['SUBJECT_ID','HADM_ID'], how='inner')
merge_pd = merge_pd.merge(labtest_pd, on=['SUBJECT_ID','HADM_ID'], how='inner')
merge_pd = merge_pd.merge(notes_pd, on=['SUBJECT_ID','HADM_ID'], how='inner')
merge_pd = merge_pd.merge(proc_pd, on=['SUBJECT_ID','HADM_ID'], how='inner')
merge_pd = merge_pd.merge(adm_pd, on=['SUBJECT_ID','HADM_ID'], how='inner')

merge_pd.to_pickle('./merge_data.pkl')

print(len(single_patient_ids), len(multi_patient_ids), len(merge_pd.SUBJECT_ID.unique()))
print("total hadm ids:{}\nsingle_hadm_ids:{}\n multi_hadm_ids:{}\n".format(
    len(merge_pd.HADM_ID.unique()), len(merge_pd[merge_pd.SUBJECT_ID.isin(single_patient_ids)].HADM_ID.unique()),
    len(merge_pd[merge_pd.SUBJECT_ID.isin(multi_patient_ids)].HADM_ID.unique())
))

total_subj = set(single_patient_ids) | set(multi_patient_ids)
merge_subj = set(merge_pd.SUBJECT_ID.unique())
intersect_subj = total_subj & merge_subj
differece_subj = total_subj.difference(merge_subj)
print(len(total_subj), len(merge_subj), len(intersect_subj), len(differece_subj))

data_single_visit = merge_pd[merge_pd.SUBJECT_ID.isin(single_patient_ids)]
data_multi_visit = merge_pd[merge_pd.SUBJECT_ID.isin(multi_patient_ids)]
data_single_visit.to_pickle('./data-single-visit.pkl')
data_multi_visit.to_pickle('./data-multi-visit.pkl')

import random
from random import shuffle
random.seed(1203)

def split_dataset(data_path='./data-multi-visit.pkl'):
    data = pd.read_pickle(data_path)
    sample_id = data['SUBJECT_ID'].unique()
    
    random_number = [i for i in range(len(sample_id))]
    shuffle(random_number)
    
    train_id = sample_id[random_number[:int(len(sample_id)*0.8)]]
    eval_id = sample_id[random_number[int(len(sample_id)*0.85): int(len(sample_id)*0.9)]]
    test_id = sample_id[random_number[int(len(sample_id)*0.9):]]
    
    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')
    
    ls2file(train_id, 'train-id.txt')
    ls2file(eval_id, 'dev-id.txt')
    ls2file(test_id, 'test-id.txt')
    
    print('train size: %d, eval size: %d, test size: %d' % (len(train_id), len(eval_id), len(test_id)))
    
split_dataset()

print("#patient {}".format(merge_pd.SUBJECT_ID.unique().shape))
print("#clincal events {}".format(len(merge_pd)))
diag = merge_pd.DIAG_CODE.values
proc = merge_pd.PRO_CODE.values
ndc = merge_pd.NDC.values
itemid = merge_pd.ITEMID.values
category = merge_pd.CATEGORY_TEXT.values
unique_diag = set([j for i in diag for j in list(i)])
unique_proc = set([j for i in proc for j in list(i)])
unique_ndc = set([k for i in ndc for j in list(i) for k in list(j)])
unique_itemid = set([k for i in itemid for j in list(i) for k in list(j)])
unique_category = set([j[0] for i in category for j in list(i)])
print('#diagnosis ', len(unique_diag))
print('#procedure ', len(unique_proc))
print('#labtest item ', len(unique_itemid))
print('#med ndc ', len(unique_ndc))
print('text category ', len(unique_category))
write_ids(unique_diag, './dx-vocab.txt')
write_ids(unique_proc, './px-vocab.txt')
write_ids(unique_itemid, './item-vocab.txt')
write_ids(unique_ndc, './rx-vocab.txt')
write_ids(unique_category, './text-category.txt')


avg_diag = 0
avg_proc = 0
avg_ndc = 0
avg_item = 0
max_diag = 0
max_proc = 0
max_item = 0
max_ndc = 0
cnt = 0
max_visit = 0
avg_visit =0
for subjid in merge_pd.SUBJECT_ID.unique():
    subj_data = merge_pd[merge_pd.SUBJECT_ID==subjid]
    x = []
    y = []
    z = []
    u = []
    visit_cnt = 0
    for idx, row in subj_data.iterrows():
        visit_cnt += 1
        cnt += 1 
        x.extend(list(row['DIAG_CODE']))
        y.extend(list(row['PRO_CODE']))
        for i in list(row['ITEMID']):
            z.extend(list(i))
        for i in list(row['NDC']):
            u.extend(list(i))
    x = set(x)
    y = set(y)
    z = set(z)
    u = set(u)
    avg_diag+=len(x)
    avg_proc+=len(y)
    avg_item+=len(z)
    avg_ndc+=len(u)
    avg_visit+=visit_cnt
    if len(x) > max_diag:
        max_diag = len(x)
    if len(y) > max_proc:
        max_proc = len(y)
    if len(z) > max_item:
        max_item = len(z)
    if len(u) > max_ndc:
        max_ndc = len(u)
    if visit_cnt > max_visit:
        max_visit = visit_cnt
print('#avg of diagnoses ', avg_diag/ cnt)
print('#avg of medicines ', avg_ndc/ cnt)
print('#avg of procedures ', avg_proc/ cnt)
print('#avg of labtest ', avg_item/ cnt)
print('#avg of vists ', avg_visit/ len(merge_pd['SUBJECT_ID'].unique()))
print('#avg of item len of labtest ', merge_pd.itemLen.sum()/cnt)
print('#avg of ndc len of med ', merge_pd.NDC_datLen.sum()/ cnt)
print('#avg of category of text ', merge_pd.TEXT_len.sum()/ cnt)

print('#max of diagnoses ', max_diag)
print('#max of medicines ', max_ndc)
print('#max of procedures ', max_proc)
print('#max of labtest', max_item)
print('#max of visit ', max_visit)
print('#max item len of labtest ', merge_pd.itemLen.max())
print('#max ndc len of medication ', merge_pd.NDC_datLen.max())
print('#max text len of text ', merge_pd.TEXT_len.max())

import pandas as pd
import numpy as np
from tqdm import tqdm
merge_data = pd.read_pickle('./merge_data.pkl')
len_items = []
len_ndcs = []
len_set_item = []
len_set_ndc = []
len_diag = []
len_proc = []
len_ep_item = []
len_ep_ndc = []
for _, row in tqdm(merge_data.iterrows(), desc='rows'):
    diag, proc, item, ndc = list(row['DIAG_CODE']), list(row['PRO_CODE']), list(row['ITEMID']), list(row['NDC'])
    x = []
    y = []
    for i in item:
        x.extend(list(i))
    for n in ndc:
        y.extend(list(n))
    len_diag.append(len(diag))
    len_proc.append(len(proc))
    len_items.append(len(item))
    len_ndcs.append(len(ndc))
    len_set_item.append(len(set(x)))
    len_set_ndc.append(len(set(y)))
    len_ep_ndc.append(len(y))
    len_ep_item.append(len(x))
print('all flatten item:{}\nall flatten ndc:{}\nmean_item:{}\tmean_ndc:{}\nset_item:{}\nset_ndc:{}\nmean_set_item:{}\tmean_set_ndc:{}\n'.format(
    np.max(len_items), np.max(len_ndcs), np.mean(len_items), np.mean(len_ndcs), \
    np.max(len_set_item), np.max(len_set_ndc), np.mean(len_set_item), np.mean(len_set_ndc), \
    ))
print("all diag:{}\nall procs:{}\nmean_diag:{}\tmean_proc:{}".format(
    np.max(len_diag), np.max(len_proc), np.mean(len_diag), np.mean(len_proc)
))
print("max extend item:{}\tmax extend ndc:{}".format(np.max(len_ep_item), np.max(len_ep_ndc)))
