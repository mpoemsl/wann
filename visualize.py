import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Visualizes statistics of a WANN experiment.")

parser.add_argument("train_experiment_folder", type=str, help="Path to a train experiment folder.")
parser.add_argument("test_experiment_folder", type=str, help="Path to a test experiment folder.")

def main():

    args = parser.parse_args()

    train_exp_name = args.train_experiment_folder.split("/")[-2]
    test_exp_name = args.test_experiment_folder.split("/")[-2]

    assert train_exp_name == test_exp_name, "Train and test folder do not belong to the same experiment."
    dataset, n_gen, pop_size, weight_type = train_exp_name.split("_")[1:-1]
    n_gen, pop_size = int(n_gen), int(pop_size)

    mean_losses, mean_n_cons, mean_n_layers = get_mean_stats(args.train_experiment_folder)
    n_gen = min(mean_losses.shape[0], n_gen) 

    eval_df = pd.read_csv(args.test_experiment_folder + "eval_scores.csv", index_col=0)
    eval_name = eval_df.columns[0]
    mean_eval_scores = eval_df[eval_name].values

    title = "{} Generations of WANN Training on {} \nwith Population Size {} and {} Weight(s)".format(n_gen, titlelize(dataset), pop_size, titlelize(weight_type))
    plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, title, "plots/{}.png".format(train_exp_name))  



def plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, title, exp_fp):

    gens = np.arange(mean_losses.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    axes[0, 0].plot(gens, mean_eval_scores, color="blue")
    axes[0, 0].set_ylabel(titlelize(eval_name))
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("Mean {} Score".format(eval_name.upper()))

    axes[0, 1].plot(gens, mean_losses, color="green")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Mean Loss")

    axes[1, 0].plot(gens, mean_n_cons, color="red")
    axes[1, 0].set_xlabel("Generations")
    axes[1, 0].set_ylabel("Connections")
    axes[1, 0].set_title("Mean Number of Connections")

    axes[1, 1].plot(gens, mean_n_layers, color="yellow")
    axes[1, 1].set_xlabel("Generations")
    axes[1, 1].set_ylabel("Layers")
    axes[1, 1].set_title("Mean Number of Layers")
    
    
    plt.suptitle(title)   

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.85, bottom=0.10)

    plt.savefig(exp_fp)
    plt.show()


def titlelize(word):

    chars = list(word)
    chars[0] = chars[0].upper()

    return "".join(chars)


def get_mean_stats(train_exp_folder):
    
    dfs = []
    gen = 0

    while os.path.exists(train_exp_folder + "stats_gen_{}.csv".format(gen)):
        dfs.append(pd.read_csv(train_exp_folder + "stats_gen_{}.csv".format(gen), index_col=0))
        gen += 1

    if gen == 0:
        raise Exception("Log not found!")

    mean_losses = np.array([df["mean_losses"].mean() for df in dfs])
    mean_n_cons = np.array([df["n_cons"].mean() for df in dfs])
    mean_n_layers = np.array([df["n_layers"].mean() for df in dfs])

    return mean_losses, mean_n_cons, mean_n_layers


if __name__ == "__main__":
    main()
