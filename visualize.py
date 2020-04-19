""" Script to visualize results of a WANN experiment. """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Visualizes statistics of a WANN experiment.")
parser.add_argument("exp_folder", type=str, help="Path to a experiment folder.")

def main(exp_folder):

    exp_name = exp_folder.split("/")[-2]
    dataset_name, n_gen, pop_size, weight_type = exp_name.split("_")[1:-1]
    n_gen, pop_size = int(n_gen), int(pop_size)

    if dataset_name == "mnist":
        sample_size = 1000
        loss_name = "cce"
    elif dataset_name == "forestfires":
        sample_size = 100
        loss_name = "mse"
    else:
        raise Exception("Invalid datset name '{}'!".format(dataset_name))

    mean_losses, mean_n_cons, mean_n_layers = get_mean_stats(exp_name)
    n_gen = min(mean_losses.shape[0], n_gen) 

    eval_df = pd.read_csv("experiments/{}/test/eval_scores.csv".format(exp_name), index_col=0)
    eval_name = eval_df.columns[0]
    mean_eval_scores = eval_df[eval_name].values

    if not os.path.exists("plots"):
        os.mkdir("plots")

    dataset_name, weight_type = titlelize(dataset_name), titlelize(weight_type)

    title = "{} Generations of WANN Training on {} \nwith Population Size {} and {} Weight(s)".format(n_gen, dataset_name, pop_size, weight_type)
    plot_fp = "plots/{}.png".format(exp_name)

    plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, loss_name, sample_size, title, plot_fp)  


def plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, loss_name, samples_per_step, title, plot_fp):

    n_samples = np.arange(mean_losses.shape[0]) * samples_per_step
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    axes[0, 0].plot(n_samples, mean_eval_scores, color="blue")
    axes[0, 0].set_ylabel(titlelize(eval_name))
    axes[0, 0].set_title("Mean {} Score".format(eval_name.upper()))

    axes[0, 1].plot(n_samples, mean_losses, color="green")
    axes[0, 1].set_ylabel(titlelize(loss_name))
    axes[0, 1].set_title("Mean {} Loss".format(loss_name.upper()))

    axes[1, 0].plot(n_samples, mean_n_cons, color="red")
    axes[1, 0].set_xlabel("Number of Observed Samples")
    axes[1, 0].set_ylabel("Connections")
    axes[1, 0].set_title("Mean Number of Connections")

    axes[1, 1].plot(n_samples, mean_n_layers, color="yellow")
    axes[1, 1].set_xlabel("Number of Observed Samples")
    axes[1, 1].set_ylabel("Layers")
    axes[1, 1].set_title("Mean Number of Hidden Layers")
    
    plt.suptitle(title)   

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.85, bottom=0.10)

    plt.savefig(plot_fp)
    plt.show()


def titlelize(word):

    chars = list(word)
    chars[0] = chars[0].upper()

    return "".join(chars)


def get_mean_stats(exp_name):
    
    dfs = []
    gen = 0

    while os.path.exists("experiments/{}/train/stats_gen_{}.csv".format(exp_name, gen)):
        dfs.append(pd.read_csv("experiments/{}/train/stats_gen_{}.csv".format(exp_name, gen), index_col=0))
        gen += 1

    if gen == 0:
        raise Exception("Log not found!")

    mean_losses = np.array([df["mean_losses"].mean() for df in dfs])
    mean_n_cons = np.array([df["n_cons"].mean() for df in dfs])
    mean_n_layers = np.array([df["n_layers"].mean() for df in dfs])

    return mean_losses, mean_n_cons, mean_n_layers


if __name__ == "__main__":

    args = parser.parse_args()
    main(args.exp_folder)
