# coding:utf-8

import os
import pandas as pd
from tqdm import tqdm


def text_clean(text):
    text = str(text)
    bad_str = ['\u200d', '\u2006', '\u202d', '\u202c', '\ufe0f', '\u0e34', '\u22ef', '\u40fc', '\ufeff',
               '\u2005', '\uf236', '\uf525', '\u20e3', '\uff65', '\u03d6', '\u2022', '\ue056', '\xa0',
               '\u2084', '\t', '\n']
    for i in bad_str:
        text = text.replace(i, '')

    return text.strip()


# process BQ
BQ_PATH = '../../data/BQ'

BQ_train_path = os.path.join(BQ_PATH, 'train')
BQ_dev_path = os.path.join(BQ_PATH, 'dev')
BQ_test_path = os.path.join(BQ_PATH, 'test')

BQ_train_df = pd.read_csv(BQ_train_path, sep='\t', header=None, engine='python', error_bad_lines=False)
BQ_test_df = pd.read_csv(BQ_test_path, sep='\t', header=None, engine='python', error_bad_lines=False)
BQ_dev_df = pd.read_csv(BQ_dev_path, sep='\t', header=None, engine='python', error_bad_lines=False)

BQ_df = pd.concat([BQ_train_df, BQ_test_df, BQ_dev_df], 0)

BQ_sent = []
for i, row in tqdm(BQ_df.iterrows(), desc='', total=len(BQ_df)):
    sentence_a, sentence_b, label = row[0], row[1], row[2]
    if sentence_a == 'nan' or sentence_b == 'nan':
        continue
    sentence_a, sentence_b = text_clean(sentence_a), text_clean(sentence_b)
    sent = sentence_a + '\t' + sentence_b + '\t' + str(label)
    BQ_sent.append(sent)


# process LCQMC
LCQMC_PATH = '../../data/LCQMC'

LCQMC_train_path = os.path.join(LCQMC_PATH, 'train')
LCQMC_dev_path = os.path.join(LCQMC_PATH, 'dev')
LCQMC_test_path = os.path.join(LCQMC_PATH, 'test')

LCQMC_train_df = pd.read_csv(LCQMC_train_path, sep='\t', header=None, engine='python', error_bad_lines=False)
LCQMC_test_df = pd.read_csv(LCQMC_test_path, sep='\t', header=None, engine='python', error_bad_lines=False)
LCQMC_dev_df = pd.read_csv(LCQMC_dev_path, sep='\t', header=None, engine='python', error_bad_lines=False)

LCQMC_df = pd.concat([LCQMC_train_df, LCQMC_test_df, LCQMC_dev_df], 0)

LCQMC_sent = []
for i, row in tqdm(LCQMC_df.iterrows(), desc='', total=len(LCQMC_df)):
    sentence_a, sentence_b, label = row[0], row[1], row[2]
    if sentence_a == 'nan' or sentence_b == 'nan':
        continue
    sentence_a, sentence_b = text_clean(sentence_a), text_clean(sentence_b)
    sent = sentence_a + '\t' + sentence_b + '\t' + str(label)
    LCQMC_sent.append(sent)

# process OPPO
OPPO_PATH = '../../data/OPPO'

OPPO_train_path = os.path.join(OPPO_PATH, 'train')
OPPO_dev_path = os.path.join(OPPO_PATH, 'dev')

OPPO_train_df = pd.read_csv(OPPO_train_path, sep='\t', header=None, engine='python', error_bad_lines=False)
OPPO_dev_df = pd.read_csv(OPPO_dev_path, sep='\t', header=None, engine='python', error_bad_lines=False)

OPPO_df = pd.concat([OPPO_train_df, OPPO_dev_df], 0)

OPPO_sent = []
for i, row in tqdm(OPPO_df.iterrows(), desc='', total=len(OPPO_df)):
    sentence_a, sentence_b, label = row[0], row[1], row[2]
    if sentence_a == 'nan' or sentence_b == 'nan':
        continue
    sentence_a, sentence_b = text_clean(sentence_a), text_clean(sentence_b)
    sent = sentence_a + '\t' + sentence_b + '\t' + str(label)
    OPPO_sent.append(sent)

# process test A
test_A_PATH = '../../data/test_A.tsv'

test_A_df = pd.read_csv(test_A_PATH, sep='\t', header=None, engine='python', error_bad_lines=False)

test_A_sent = []
for i, row in tqdm(test_A_df.iterrows(), desc='', total=len(test_A_df)):
    sentence_a, sentence_b = row[0], row[1]
    if sentence_a == 'nan' or sentence_b == 'nan':
        continue
    sentence_a, sentence_b = text_clean(sentence_a), text_clean(sentence_b)
    sent = sentence_a + '\t' + sentence_b
    test_A_sent.append(sent)

all_sent = BQ_sent + LCQMC_sent + OPPO_sent

process_root_path = '../../user_data/process_data'
os.makedirs(process_root_path, exist_ok=True)
BQ_train_data_path = os.path.join(process_root_path, 'BQ_train.txt')
LCQMC_train_data_path = os.path.join(process_root_path, 'LCQMC_train.txt')
OPPO_train_data_path = os.path.join(process_root_path, 'OPPO_train.txt')
pretrain_data_path = os.path.join(process_root_path, 'pretrain.txt')
train_data_path = os.path.join(process_root_path, 'train.txt')
test_data_path = os.path.join(process_root_path, 'test.txt')

# write to txt
with open(pretrain_data_path, 'w', encoding='utf-8') as f:
    for i in all_sent:
        sent_a, sent_b, label = i.strip().split('\t')
        if sent_a == 'nan' or sent_b == 'nan':
            print('nan')
        else:
            pretrain_sent = sent_a + '\t' + sent_b
            f.writelines(pretrain_sent + '\n')
    for i in test_A_sent:
        sent_a, sent_b = i.strip().split('\t')
        pretrain_sent = sent_a + '\t' + sent_b
        f.writelines(pretrain_sent + '\n')

with open(BQ_train_data_path, 'w', encoding='utf-8') as f:
    for i in BQ_sent:
        sent_a, sent_b, label = i.strip().split('\t')
        if sent_a == 'nan' or sent_b == 'nan':
            print('nan')
        else:
            train_sent = sent_a + '\t' + sent_b + '\t' + label
            f.writelines(train_sent + '\n')

with open(LCQMC_train_data_path, 'w', encoding='utf-8') as f:
    for i in LCQMC_sent:
        sent_a, sent_b, label = i.strip().split('\t')
        if sent_a == 'nan' or sent_b == 'nan':
            print('nan')
        else:
            train_sent = sent_a + '\t' + sent_b + '\t' + label
            f.writelines(train_sent + '\n')

with open(OPPO_train_data_path, 'w', encoding='utf-8') as f:
    for i in OPPO_sent:
        sent_a, sent_b, label = i.strip().split('\t')
        if sent_a == 'nan' or sent_b == 'nan':
            print('nan')
        else:
            train_sent = sent_a + '\t' + sent_b + '\t' + label
            f.writelines(train_sent + '\n')


with open(train_data_path, 'w', encoding='utf-8') as f:
    for i in all_sent:
        sent_a, sent_b, label = i.strip().split('\t')
        if sent_a == 'nan' or sent_b == 'nan':
            print('nan')
        else:
            train_sent = sent_a + '\t' + sent_b + '\t' + label
            f.writelines(train_sent + '\n')

with open(test_data_path, 'w', encoding='utf-8') as f:
    for i in test_A_sent:
        sent_a, sent_b = i.strip().split('\t')
        test_sent = sent_a + '\t' + sent_b
        f.writelines(test_sent + '\n')
