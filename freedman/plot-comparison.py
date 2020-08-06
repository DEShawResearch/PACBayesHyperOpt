import argparse
import numpy as np
import pickle
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import namedtuple
DataSet = namedtuple("DataSet", [
    "train_data",
    "valid_data",
    "test_data",
    "train_labels",
    "valid_labels",
    "test_labels",
])

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
sns.set_style("white")
sns.set_style("ticks")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data-set",
        default="freedman-data-set.pkl",
        help="Path to freedman data set file created by other script")
    p.add_argument(
        "--naive-results",
        default="freedman-naive-results.jsonl",
        help="Resuts from the naive script",
    )
    p.add_argument(
        "--results",
        default="freedman-results.jsonl",
        help="Resuts from the freedman script",
    )
    args = p.parse_args()

    with open(args.data_set, "rb") as fp:
        data_set = pickle.load(fp)

    naive_results = load_jsonl(args.naive_results)
    our_results = load_jsonl(args.results)

    plot_unregularized_results(naive_results, data_set)
    plot_regularized_results(our_results, data_set)

def plot_unregularized_results(results, data_set):
    test_errors = []
    top_objectives = []
    for result in results:
        index_set = result["top_index_set"]

        coefs, _, _, _ = np.linalg.lstsq(data_set.train_data[:, index_set], data_set.train_labels, rcond=None)
        test_error = get_error(coefs, data_set.test_data[:, index_set], data_set.test_labels)
        valid_error = get_error(coefs, data_set.valid_data[:, index_set], data_set.valid_labels)

        test_errors.append(test_error)
        top_objectives.append(valid_error)

    fig, ax1 = plt.subplots(figsize=(7,5))
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='y', which='major', labelsize=14, labelcolor='k')
    ax1.plot(np.arange(1, len(results) + 1), test_errors, linewidth=3, label='Test Fit', color='k')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(results) + 1), top_objectives, linewidth=3, label='Objective', color='b', ls='dashed')
    ax1.set_ylabel('Test Set MSE', size=16)
    ax2.tick_params(axis='y', which='major', labelsize=14, labelcolor='b')
    ax2.set_ylabel('Hyperparameter Objective', size=16)
    ax1.set_xlabel('Number of Predictors Selected', size=16)
    plt.tight_layout()
    plt.savefig("unregularized_objective.pdf")


def plot_regularized_results(results, data_set):
    for result in results:
        test_errors = []
        top_objectives = []
        num_chains = len(results[-1]["top_objective_list"])
        for r_dict in results:
            top_index_set = r_dict["top_index_set"]
            coefs, resids, rank, S = np.linalg.lstsq(data_set.train_data[:,top_index_set], data_set.train_labels, rcond=None)
            top_objectives.append(r_dict["top_objective"])
            test_error = get_error(coefs, data_set.test_data[:, top_index_set], data_set.test_labels)
            test_errors.append(test_error)

    # collect into DataFrame for easy error bar calculation
    df_dict = {}
    df_dict["Number of Predictors Selected"] = [i + 1 for i in range(len(results)) for j in range(num_chains)]
    df_dict["Hyperparameter Objective"] = []
    for idx, r in enumerate(results):
        df_dict["Hyperparameter Objective"].extend(r["top_objective_list"])
    df = pd.DataFrame(df_dict)

    # plotting code
    plt.figure(figsize=(7,5))
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.tick_params(axis="both", which="major", labelsize=14)
    ax1 = sns.lineplot(np.arange(1, len(results)+1), test_errors, linewidth=3, color="k")
    ax1.set_ylabel("Test Set MSE", size=16)
    ax1.tick_params(axis="y", which="major", labelsize=14, labelcolor="k")
    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", which="major", labelsize=14, labelcolor="b")
    sns.lineplot(data=df, x="Number of Predictors Selected", y="Hyperparameter Objective", color="b", ax=ax2)
    ax2.lines[0].set_linestyle("--")
    ax1.lines[0].set_linewidth(3)
    ax2.lines[0].set_linewidth(3)
    ax2.set_ylabel("Hyperparameter Objective", size=16)
    ax1.set_xlabel("Number of Predictors Selected", size=16)
    plt.tight_layout()
    plt.savefig("regularized_objective.pdf")


def load_jsonl(path):
    data = []
    with open(path) as fp:
        for line in fp:
            if not line.startswith("#"):
                data.append(json.loads(line))
    return data


def get_error(coefs, X, labels):
    """Compute the mean squared error"""
    preds = np.dot(X, coefs.reshape(-1, 1)).reshape(-1)
    MSE = np.mean((preds - labels)**2)
    return MSE



if __name__ == "__main__":
    main()
