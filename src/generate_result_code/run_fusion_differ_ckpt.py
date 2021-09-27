# coding:utf-8

import os
import sys
import csv
import numpy as np
import pandas as pd

sys.path.append('../../src')
from argparse import ArgumentParser


def fusion(args):
    model_name = args.ckpt_name.split(',')
    weights = list(map(float, args.weights.split(',')))
    assert len(model_name) == len(weights), "models must equal weights !!!"
    # assert sum(weights) == 1, "weights sum must equal 1 !!!"
    k = 0
    predictions = 0
    for i, name in enumerate(model_name):
        try:
            logit_path = os.path.join(args.result_path, 'output_result', name, 'full_logit.csv')
            tmp = pd.read_csv(logit_path)
            tmp = tmp.values * weights[i]
            predictions += tmp
        except:
            print("fail load %s result!!!" % name)

    predictions = np.argmax(predictions, axis=-1)
    predictions = [str(i) for i in predictions]
    write2tsv(args.submit_path, predictions)


def write2tsv(output_path, data):
    with open(output_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerows(data)


def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default="0.3,0.3,0.4")
    parser.add_argument('--ckpt_name', type=str, default='BQ,LCQMC,OPPO')
    parser.add_argument('--result_path', type=str, default="../../user_data")
    parser.add_argument('--test_path', type=str, default='../../data/test.csv')
    parser.add_argument('--submit_path', type=str, default=f'../../ccf_qianyan_qm_result_A.csv')

    args = parser.parse_args()

    fusion(args)


if __name__ == '__main__':
    main()
