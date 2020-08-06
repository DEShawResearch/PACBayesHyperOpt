#!/usr/bin/env python3
import numpy as np
import pickle
import json
import argparse
from typing import List, Tuple, Optional
from collections import namedtuple

DataSet = namedtuple(
    "DataSet",
    [
        "train_data",
        "valid_data",
        "test_data",
        "train_labels",
        "valid_labels",
        "test_labels",
    ],
)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lr", type=float, help="learning rate", default=0.1)
    p.add_argument("--beta-min", type=float, help="min beta", default=0.1)
    p.add_argument("--beta-max", type=float, help="max beta", default=0.1)
    p.add_argument(
        "--beta-num",
        type=int,
        help="number of betas to test (linear between min and max)",
        default=1,
    )
    p.add_argument("--num-chains", type=int, help="number of chains", default=10)
    p.add_argument(
        "--num-indices", type=int, help="number of indices to select", default=20
    )
    p.add_argument(
        "--num-inner-steps", type=int, help="number of steps per chain", default=50
    )
    p.add_argument(
        "--lr-div-factor",
        type=float,
        default=4,
        help="fisher chain error is (lr/lr_div_factor) * np.linalg.norm(grad - v_grad)**2)",
    )
    p.add_argument(
        "--null",
        action="store_true",
        default=False,
        help="whether there should be 2 true predictors included in the dataset",
    )
    p.add_argument(
        "--output-results",
        default="freedman-results.jsonl",
        help="Path to save results",
    )
    args = p.parse_args()

    indices_set = []

    #
    # create_dataset will dump dataset generated
    # for experiment to freedman-data-set.pkl
    # in the directory that this script is called from
    #
    data_set = create_dataset(args.null)
    trial_betas = np.linspace(args.beta_min, args.beta_max, num=args.beta_num)

    with open(args.output_results, "w") as fp:
        json_dump = (
            json.dumps(
                {
                    "lr": args.lr,
                    "lr_div_factor": args.lr_div_factor,
                    "trial_betas": trial_betas.tolist(),
                    "num_chains": args.num_chains,
                    "num_inner_steps": args.num_inner_steps,
                },
                sort_keys=True,
            )
            + "\n"
        )
        fp.write("# " + json_dump)

    for i in range(args.num_indices):
        print("Forward selection iteration %d..." % i)
        indices_set = forward_selection(
            current_indices_set=indices_set,
            data_set=data_set,
            lr=args.lr,
            lr_div_factor=args.lr_div_factor,
            betas=trial_betas,
            num_chains=args.num_chains,
            num_inner_steps=args.num_inner_steps,
            output_results=args.output_results,
        )


def create_dataset(null: bool = False) -> DataSet:
    """Create the Freedman's paradox dataset

    There are 500 points and 500 features. If null == True
    the features are all normally distributed random values,
    as are the labels. If null == False, the first two features
    are valid predictors and the others features are as before.

    Dumps dataset to freedman-data-set.pkl which is then used by 
    freedman_naive.py to do feature selection on an identical data 
    set without regularization.
    """
    random = np.random.RandomState(0)
    num_points = 500
    num_preds = 500
    split = 250
    data = random.normal(0, 1, (num_points, num_preds))
    if null:
        labels = random.normal(0, 1, (num_points,))
    else:
        labels = data[:, 0] + data[:, 1] + random.normal(0, 2, (num_points,))
        labels = labels / np.sqrt(6)

    train_data = data[0:split, :]
    train_labels = labels[0:split]

    valid_data = data[split:, :]
    valid_labels = labels[split:]

    test_data = random.normal(0, 1, (num_points, num_preds))
    test_labels = test_data[:, 0] + test_data[:, 1] + random.normal(0, 2, (num_points,))
    test_labels = test_labels / np.sqrt(6)

    data_set = DataSet(
        train_data, valid_data, test_data, train_labels, valid_labels, test_labels
    )

    with open("freedman-data-set.pkl", "wb") as f:
        pickle.dump(data_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_set


def forward_selection(
    current_indices_set: List[int],
    data_set: DataSet,
    lr: float,
    lr_div_factor: float,
    betas: List[float],
    num_chains: int,
    num_inner_steps: int,
    output_results: str,
) -> List[int]:
    """Perform one step of forward selection to determine the additional feature
    to add to the model. This function screens each of the remaining features
    not currently included in current_indices_set and greedily selects one additional
    feature.

    Args:
        current_indices_set (List[int]) the features currently included in the model
        data_set (DataSet) training and validation data
        lr (float) learning rate
        lr_div_factor (float) scaling factor that determines the regularization penalty
        beta (float) parameter of the sampled Gibbs posterior
        num_chains (int) the number of chains to run
        num_inner_steps (int) the number of steps of inner optimization to run

    Returns:
        new_indices_set (List[int]) the new set of included features, equal to current_indices_set
            plus one new one
    """
    top_objective = float("inf")
    top_index_set: Optional[List[int]] = None
    top_error: Optional[float] = None
    top_r2: Optional[float] = None
    top_beta: Optional[float] = None
    top_objective_list: List[float] = []

    remaining_indices = [
        i for i in range(data_set.train_data.shape[1]) if i not in current_indices_set
    ]

    for beta in betas:
        for index in remaining_indices:
            trial_index_set = current_indices_set + [index]
            objective, error, r2, objective_list = run_chains(
                lr=lr,
                lr_div_factor=lr_div_factor,
                beta=beta,
                data_set=data_set,
                index_set=trial_index_set,
                num_chains=num_chains,
                num_inner_steps=num_inner_steps,
            )
            if objective < top_objective:
                top_objective = objective
                top_index_set = trial_index_set
                top_error = error
                top_r2 = r2
                top_beta = beta
                top_objective_list = objective_list
    assert top_index_set is not None
    assert top_error is not None
    assert top_r2 is not None
    assert top_beta is not None

    coefs, _resids, _rank, _S = np.linalg.lstsq(
        data_set.train_data[:, top_index_set], data_set.train_labels, rcond=None
    )
    true_r2 = get_r2(coefs, data_set.test_data[:, top_index_set], data_set.test_labels)
    valid_r2 = get_r2(
        coefs, data_set.valid_data[:, top_index_set], data_set.valid_labels
    )

    bound_r2 = 1 - (len(data_set.valid_labels) * top_objective) / (
        len(data_set.valid_labels) * np.var(data_set.valid_labels)
    )

    print("Forward selection iteration complete")
    print(f"Top Objective: {top_objective}")
    print(f"Top Index Set: {top_index_set}")
    print(f"Top Error: {top_error}")
    print(f"Beta {top_beta}")
    print(f"Langevin R-squared: {top_r2}")
    print(f"Valid R-squared: {valid_r2}")
    print(f"Bound R-squared: {bound_r2}")
    print(f"True R-squared: {true_r2}")
    print()

    with open(output_results, "a") as fp:
        fp.write(
            json.dumps(
                {
                    "num_selected": len(top_index_set),
                    "top_r2": top_r2,
                    "top_objective": top_objective,
                    "top_error": top_error,
                    "top_index_set": top_index_set,
                    "top_beta": top_beta,
                    "top_objective_list": top_objective_list,
                },
                sort_keys=True,
            )
            + "\n"
        )

    return top_index_set


def run_chains(
    lr: float,
    lr_div_factor: float,
    beta: float,
    data_set: DataSet,
    index_set: List[int],
    num_chains: int,
    num_inner_steps: int,
) -> Tuple[float, float, float, List[float]]:
    """Runs multiple chains of optimization to estimate the objective
    derived in Eq. 7. This function runs Langevin Dynamics specified by the
    input parameters to fit the model whose features are determined by
    'index_set' to the training data set.

    Args:
        lr (float) learning rate
        lr_div_factor (float) scaling factor that determines the regularization penalty
        beta (float) parameter of the sampled Gibbs posterior
        data_set (DataSet) training and validation data
        index_set (List[int]) the to include in the model
        num_chains (int) the number of chains to run
        num_inner_steps (int) the number of steps of inner optimization to run

    Returns:
        objective (float) average objective over the chains
        error (float) mean squared error averaged over the chains
        r2 (float) average r^2 over the chains
        chain_objectives (List[float]) objective of each individual chain
    """

    random = np.random.RandomState(0)

    X = data_set.train_data[:, index_set]
    vX = data_set.valid_data[:, index_set]

    chain_objective = 0.0
    chains = [random.normal(size=(len(index_set), 1)) for i in range(num_chains)]
    chain_objectives = [0.0 for i in range(num_chains)]
    for _step in range(num_inner_steps):
        fisher_chains = []
        for chain in range(num_chains):
            grad = gradient(chains[chain], X, data_set.train_labels)
            v_grad = gradient(chains[chain], vX, data_set.valid_labels)
            noise = random.normal(
                scale=np.sqrt(2 * lr / (beta * X.shape[0])), size=grad.shape
            )
            chains[chain] = chains[chain] - lr * grad + noise
            fisher_chains.append(
                beta * (lr / lr_div_factor) * np.linalg.norm(grad - v_grad) ** 2
            )
            chain_objectives[chain] += fisher_chains[-1]
        chain_objective += np.mean(fisher_chains)

    chain_objective = np.sqrt(chain_objective)
    chain_objectives = [np.sqrt(co) for co in chain_objectives]
    val_errors = []
    val_r2s = []
    avg_error = 0.0
    avg_r2 = 0.0
    for idx, chain in enumerate(chains):
        val_error = get_error(chain, vX, data_set.valid_labels)
        chain_objectives[idx] += val_error
        val_r2 = get_r2(chain, vX, data_set.valid_labels)
        val_errors.append(val_error)
        val_r2s.append(val_r2)

    avg_error = np.mean(val_errors)
    avg_r2 = np.mean(val_r2s)

    chain_objective += avg_error
    return chain_objective, avg_error, avg_r2, chain_objectives


def get_error(coefs, X, labels):
    """Compute the mean squared error"""
    preds = np.dot(X, coefs.reshape(-1, 1)).reshape(-1)
    MSE = np.mean((preds - labels) ** 2)
    return MSE


def get_r2(coefs, X, labels):
    """Compute the R^2 statistic"""
    preds = np.dot(X, coefs.reshape(-1, 1)).reshape(-1)
    SSE = np.sum((preds - labels) ** 2)
    r_sq = 1 - SSE / (len(labels) * np.var(labels))
    return r_sq


def gradient(coefs, X, labels, clip=np.inf):
    """Compute the deriatives of the mean squared error with respect
    to the coefficients"""
    gradient = X.T @ X @ coefs.reshape(-1, 1) - X.T @ labels.reshape(-1, 1)
    gradient = gradient / len(labels)
    if np.linalg.norm(gradient) > clip:
        gradient = gradient / (np.linalg.norm(gradient) / clip)
    return gradient


if __name__ == "__main__":
    main()
