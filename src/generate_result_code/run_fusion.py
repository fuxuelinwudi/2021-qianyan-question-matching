# coding:utf-8

import os
import sys
import csv
import numpy as np
import pandas as pd

sys.path.append('../../src')
from argparse import ArgumentParser


def fusion(args):
    predictions = 0

    tmp = pd.read_csv(os.path.join(args.result_path, 'output_result', 'full_logit.csv'))
    tmp = tmp.values
    predictions += tmp
    predictions = np.argmax(predictions, axis=-1).tolist()

    predictions = [str(i) for i in predictions]
    write2tsv(args.submit_path, predictions)


def write2tsv(output_path, data):
    with open(output_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerows(data)


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, default="../../user_data")
    parser.add_argument('--test_path', type=str, default='../../data/test.csv')
    parser.add_argument('--submit_path', type=str, default=f'../../ccf_qianyan_qm_result_A.csv')

    args = parser.parse_args()

    fusion(args)


if __name__ == '__main__':
    main()
