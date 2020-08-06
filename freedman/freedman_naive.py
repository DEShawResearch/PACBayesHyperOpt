#!/usr/bin/env python3
import numpy as np
import pickle
import json
import argparse
from typing import List, Tuple, Optional
from collections import namedtuple
DataSet = namedtuple("DataSet", [
    "train_data",
    "valid_data",
    "test_data",
    "train_labels",
    "valid_labels",
    "test_labels",
])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-indices", type=int, help="number of indices to select", default=40)
    p.add_argument(
        "--data-set",
        default="freedman-data-set.pkl",
        help="Path to freedman data set file created by running freedman.py")
    p.add_argument(
        "--output-results",
        default="freedman-naive-results.jsonl",
        help="Path to save results",
    )
    args = p.parse_args()

    with open(args.data_set, 'rb') as fp:
        data_set = pickle.load(fp)

    multiple_forward_selection(data_set, args.num_indices, args.output_results)


def multiple_forward_selection(data_set, num_indices, output_results):
    index_set = []
    t_errors = []
    v_errors = []
    top_criteria = []
    for i in range(num_indices):
        print("Forward selection iteration %d..." % i)
        index, t_error, v_error, top_crit = forward_selection(data_set, index_set)
        index_set.append(index)
        t_errors.append(t_error)
        v_errors.append(v_error)
        top_criteria.append(top_crit)
        print('Model selected')
        print(f'Validation Error: {v_error}')
        print(f'Top Index Set: {index_set}')
        print()

        with open(output_results, "a") as fp:
            fp.write(
                json.dumps(
                    {
                        "num_selected": len(index_set),
                        "top_index_set": index_set,
                        "top_objective": v_error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    return index_set, t_errors, v_errors, top_criteria


def forward_selection(data_set, indices_set=[]):
    train_data = data_set.train_data
    train_labels = data_set.train_labels
    valid_data = data_set.valid_data
    valid_labels = data_set.valid_labels

    remaining_indices = [i for i in range(train_data.shape[1]) if i not in indices_set]

    top_train_error = 0.0
    top_valid_error = 0.0
    top_criterion = float("inf")
    selected_index = -1
    for rem_index in remaining_indices:
        index_set = indices_set + [rem_index]
        X = train_data[:, index_set]
        vX = valid_data[:, index_set]
        coefs, _, _, _ = np.linalg.lstsq(X, train_labels, rcond=None)
        train_error = get_error(coefs, X, train_labels)
        valid_error = get_error(coefs, vX, valid_labels)

        criterion = valid_error
        if criterion < top_criterion:
            top_valid_error = valid_error
            top_train_error = train_error
            top_criterion = criterion
            selected_index = rem_index
    return selected_index, top_train_error, top_valid_error, top_criterion


def get_error(coefs, X, labels):
    """Compute the mean squared error"""
    preds = np.dot(X, coefs.reshape(-1, 1)).reshape(-1)
    MSE = np.mean((preds - labels)**2)
    return MSE


def get_r2(coefs, X, labels):
    """Compute the R^2 statistic"""
    preds = np.dot(X, coefs.reshape(-1, 1)).reshape(-1)
    SSE = np.sum((preds - labels)**2)
    r_sq = 1 - SSE / (len(labels) * np.var(labels))
    return r_sq


if __name__ == "__main__":
    main()
