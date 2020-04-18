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

    train_exp_name = args.train_experiment_folder.split("/")[-1]
    test_exp_name = args.test_experiment_folder.split("/")[-1]

    assert train_exp_name == test_exp_name, "Train and test folder do not belong to the same experiment."
    exp_name = train_exp_name

    mean_mean_losses, mean_n_cons, mean_n_layers = get_mean_stats(args.train_experiment_folder)
    # eval_scores = pd.read_csv(args.

    gens = np.arange(gen)


    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    axes[0, 0].plot(gens, mean_eval_scores, color="blue")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Mean Evaluation Score")

    axes[0, 1].plot(gens, mean_mean_losses, color="green")
    axes[0, 1].set_ylabel("Cross-Entropy")
    axes[0, 1].set_title("Mean Loss")

    axes[1, 0].plot(gens, mean_n_cons, color="red")
    axes[1, 0].set_xlabel("Generations")
    axes[1, 0].set_ylabel("Connections")
    axes[1, 0].set_title("Mean Number of Connections")

    axes[1, 1].plot(gens, mean_n_layers, color="yellow")
    axes[1, 1].set_xlabel("Generations")
    axes[1, 1].set_ylabel("Layers")
    axes[1, 1].set_title("Mean Number of Layers")
    
    plt.suptitle("Statistics about {} Generations of Evolution in {}".format(gen, FOLDER))
    plt.tight_layout(pad=3.0)    

    plt.show()


def get_mean_stats(train_exp_folder):
    
    dfs = []
    gen = 0

    while os.path.exists("{}/gen_{}.csv".format(train_exp_folder, gen)):
        dfs.append(pd.read_csv("{}/gen_{}.csv".format(train_exp_folder, gen), index_col=0))
        gen += 1

    if gen == 0:
        raise Exception("Log not found!")

    mean_mean_losses = [df["mean_losses"].mean() for df in dfs]
    mean_n_cons = [df["n_cons"].mean() for df in dfs]
    mean_n_layers = [df["n_layers"].mean() for df in dfs]

    return mean_mean_losses, mean_n_cons, mean_n_layers


if __name__ == "__main__":
    main()
