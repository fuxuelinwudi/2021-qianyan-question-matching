# coding:utf-8

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pretrain_path = '../../../user_data/process_data/pretrain.txt'
BQ_train_path = '../../../user_data/process_data/BQ_train.txt'
LCQMC_train_path = '../../../user_data/process_data/LCQMC_train.txt'
OPPO_train_path = '../../../user_data/process_data/OPPO_train.txt'
test_A_path = '../../../user_data/process_data/test.txt'

BQ_text, LCQMC_text, OPPO_text, test_A_text = [], [], [], []
with open(BQ_train_path, 'r', encoding='utf-8')as f:
    for line_id, line in enumerate(f):
        sent_a, sent_b, label = line.strip().split('\t')
        BQ_text.append(sent_a+sent_b)
with open(LCQMC_train_path, 'r', encoding='utf-8')as f:
    for line_id, line in enumerate(f):
        sent_a, sent_b, label = line.strip().split('\t')
        LCQMC_text.append(sent_a+sent_b)
with open(OPPO_train_path, 'r', encoding='utf-8')as f:
    for line_id, line in enumerate(f):
        sent_a, sent_b, label = line.strip().split('\t')
        OPPO_text.append(sent_a+sent_b)
with open(test_A_path, 'r', encoding='utf-8')as f:
    for line_id, line in enumerate(f):
        sent_a, sent_b = line.strip().split('\t')
        test_A_text.append(sent_a+sent_b)

print('\n>> BQ: {}, LCQMC: {}, OPPO: {}, test_A: {}'.
      format(len(BQ_text), len(LCQMC_text), len(OPPO_text), len(test_A_text)))

BQ_text_len, LCQMC_text_len, OPPO_text_len, test_A_text_len = [], [], [], []
for i in BQ_text:
    BQ_text_len.append(len(i))
for i in LCQMC_text:
    LCQMC_text_len.append(len(i))
for i in OPPO_text:
    OPPO_text_len.append(len(i))
for i in test_A_text:
    test_A_text_len.append(len(i))

print('\n>> BQ - max : {}, min : {}, avg : {}'.
      format(max(BQ_text_len), min(BQ_text_len), sum(BQ_text_len)/len(BQ_text_len)))
print('\n>> LCQMC - max : {}, min : {}, avg : {}'.
      format(max(LCQMC_text_len), min(LCQMC_text_len), sum(LCQMC_text_len)/len(LCQMC_text_len)))
print('\n>> OPPO - max : {}, min : {}, avg : {}'.
      format(max(OPPO_text_len), min(OPPO_text_len), sum(OPPO_text_len)/len(OPPO_text_len)))
print('\n>> test A - max : {}, min : {}, avg : {}'.
      format(max(test_A_text_len), min(test_A_text_len), sum(test_A_text_len)/len(test_A_text_len)))


def statistic_arr(myarr, bins):
    statis = np.arange(bins.size)
    result = []
    for i in range(0, bins.size):
        statis[i] = myarr[myarr < bins[i]].size
        str_item = ("sequence length < " + str(bins[i]), str(round(statis[i] / myarr.size, 5)))
        result.append(str_item)

    print(result)


# 测试代码
BQ_arr = np.array(BQ_text_len)
BQ_bins = np.array([32, 64, 128, 192])
statistic_arr(BQ_arr, BQ_bins)

LCQMC_arr = np.array(LCQMC_text_len)
LCQMC_bins = np.array([32, 64, 128, 192])
statistic_arr(LCQMC_arr, LCQMC_bins)

OPPO_arr = np.array(OPPO_text_len)
OPPO_bins = np.array([32, 64, 128, 192])
statistic_arr(OPPO_arr, OPPO_bins)

test_A_arr = np.array(test_A_text_len)
test_A_bins = np.array([32, 64, 128, 192])
statistic_arr(test_A_arr, test_A_bins)


# plt.plot(text_len)
# plt.show()
