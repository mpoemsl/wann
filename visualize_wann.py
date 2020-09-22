""" Script to visualize results of a WANN experiment. """

from src.utilities import titlelize, plot_stats

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Visualizes statistics of a WANN experiment.")
parser.add_argument("exp_folder", type=str, help="Path to WANN experiment folder.")

def main(exp_folder):
    """ Visualizes statistics about a WANN experiment. """

    exp_name = exp_folder.split("/")[-2]
    dataset_name, n_gen, pop_size, weight_type = exp_name.split("_")[1:-1]
    n_gen, pop_size = int(n_gen), int(pop_size)

    if dataset_name == "mnist":
        sample_size = 1000
        loss_name = "cce"
        sample_limit = 30_000
    elif dataset_name == "forestfires":
        sample_size = 100
        loss_name = "mse"
        sample_limit = 20_000
    else:
        raise Exception("Invalid datset name '{}'!".format(dataset_name))

    mean_losses, mean_n_cons, mean_n_layers = get_mean_stats(exp_name)
    n_gen = min(mean_losses.shape[0], n_gen) 

    if not os.path.exists("experiments/{}/test/eval_scores.csv".format(exp_name)):
        raise Exception("Evaluation scores for {} not found, please run test.py first.".format(exp_name))

    eval_df = pd.read_csv("experiments/{}/test/eval_scores.csv".format(exp_name), index_col=0)
    eval_name = eval_df.columns[0]
    mean_eval_scores = eval_df[eval_name].values

    if not os.path.exists("plots"):
        os.mkdir("plots")

    dataset_name, weight_type = titlelize(dataset_name), titlelize(weight_type)

    title = "{} Generations of WANN Training on {} \nwith Population Size {} and {} Weight(s)".format(n_gen, dataset_name.upper(), pop_size, weight_type)
    plot_fp = "plots/wann_{}.png".format(exp_name.split("_", maxsplit=1)[1])

    plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, loss_name, sample_size, title, plot_fp, sample_limit)  


def get_mean_stats(exp_name):
    """ Extracts mean statistics from train subfolder of an experiment. """
    
    dfs = []
    gen = 0

    while os.path.exists("experiments/{}/train/stats_gen_{}.csv".format(exp_name, gen)):
        dfs.append(pd.read_csv("experiments/{}/train/stats_gen_{}.csv".format(exp_name, gen), index_col=0))
        gen += 1

    if gen == 0:
        raise Exception("Experiment data not found!")

    mean_losses = np.array([df["mean_losses"].mean() for df in dfs])
    mean_n_cons = np.array([df["n_cons"].mean() for df in dfs])
    mean_n_layers = np.array([df["n_layers"].mean() for df in dfs])

    return mean_losses, mean_n_cons, mean_n_layers


if __name__ == "__main__":

    args = parser.parse_args()
    main(args.exp_folder)
