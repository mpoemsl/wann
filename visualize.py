import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# TO-DO: 
# adapt to updated filestructure
# include accuracy from log/test
# take command-line arguments via argparse

FOLDER = "shared_weight" # "shared_weight"

def main():

    dfs = []
    gen = 0

    while os.path.exists("log/{}/gen_{}.csv".format(FOLDER, gen)):
        dfs.append(pd.read_csv("log/{}/gen_{}.csv".format(FOLDER, gen), index_col=0))
        gen += 1

    if gen == 0:
        raise Exception("Log not found!")

    gens = np.arange(gen)

    mean_eval_scores = [df["eval_scores"].mean() for df in dfs]
    mean_mean_losses = [df["mean_losses"].mean() for df in dfs]
    mean_n_cons = [df["n_cons"].mean() for df in dfs]
    mean_n_layers = [df["n_layers"].mean() for df in dfs]

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



if __name__ == "__main__":
    main()
