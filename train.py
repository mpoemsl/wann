from genetic_algorithm import evolve_population
from utilities import load_dataset
from individuum import Individuum


from sklearn.metrics import log_loss
from scipy.special import softmax
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

# initialization of hyper parameters
hyper = {
    "n_gen": 3,            # number of generations: paper does 4096
    "pop_size": 64,        # size of population: paper does 960
    "init_activation": 1,  # ReLU, MNIST specific
    "ratio_enabled": 0.05, # probability of connection being enabled when individuum is initialized
    "cull_ratio": 0.2,     # percentage of unfittest individuals who get excluded from breeding
    "elite_ratio": 0.2,    # percentage of fittest individuals who pass on to the new population unchanged
    "prob_crossover": 0.0, # percentage of how often no crossover takes place, but the best genome is passed on after mutating
    "n_cross_points": 4,   # number of crossing points in crossover
    "prob_add_node": 0.25, # probability of adding a node as mutation
    "prob_add_con": 0.25,  # probability of adding a connection as mutation
    "prob_change_activation": 0.5, # probability of changing the activation function as mutation
    "weight_values": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], # single shared weight values of the network
    "weight_type": "random",# weights can either be shared or random
    "n_rollouts": 1,        # number of repetitions when evaluation performance of a network and its weight values
    "rank_prob": 0.8,       # probability that ranking is performed via number of connections and mean loss (instead of via mean loss and min loss)
    "tournament_size": 32,  # number of individuals competing to become parent of new kid
    "tau": 0.5,             # parameter that balances off the ranking between number of connection and mean loss
    "phi": 0.5,             # parameter that balances off the ranking between min loss and mean loss
    "dataset_name": "mnist"
}


def main(n_gen=5, dataset_name="mnist", **hyper):
    """ Main loop for the WANNs construction.
    Creates a population of WANNs and evolves it over several generations.

    Parameters:
    n_gen        -  (int) number of generations or how often the evolution should take place
    dataset_name -  (string) name of the dataset for which the WANN is constructed 
    """

    hyper["experiment_name"] = "experiment_{}_{}_{}".format(n_gen, hyper["pop_size"], hyper["weight_type"])

    print("Creating folders for experiment '{}' ...".format(hyper["experiment_name"]))
    os.mkdir("best_individuums/" + hyper["experiment_name"])
    os.mkdir("log/train/" + hyper["experiment_name"])

    print("Loading MNIST training data ...")
    X, y = load_dataset(dataset_name)
    hyper["n_inputs"], hyper["n_outputs"] = X.shape[1], y.shape[1]

    print("Initializing Population ...")

    # initialize the population
    population = init_population(**hyper)

    for gen in range(n_gen):

        print("\nGeneration", gen + 1)
        start = time.time()

        print("Sampling data ...")
        inputs, targets = sample_data(X, y, hyper["n_rollouts"])

        print("Evaluating population ... ")
        # evaluate the performance of population
        eval_scores, gen_statistics = evaluate_population(population, inputs, targets, **hyper)  

        print("Evolving population ...")
        # create new population based on evaluation
        population = evolve_population(population, eval_scores, gen=gen, **hyper)  

        print("Generation lasted {:4f} seconds.".format(time.time() - start))

        # Save statistics in log
        pd.DataFrame(gen_statistics).to_csv("log/train/{}/stats_gen_{}".format(hyper["experiment_name"], gen))

    print("Finished running {} generations.".format(n_gen))


def init_population(pop_size=20, **hyper):
    """ Initializes the population by creating a lot of new individuals.

    Parameters:
    pop_size    -   (int) size of the population  
    """
    population = []
    for _ in range(pop_size):
        indiv = Individuum(**hyper)
        population.append(indiv)

    return np.array(population)


def evaluate_population(population, inputs, targets, weight_values=[], rank_prob=0.8, tau=0.5, phi=0.5, **hyper):
    """ Evaluates a complete population via mean loss and either number of connections XOR minimum loss performance.
    (Attention! Paper uses pareto dominance ranking, this is not done here! A simple scoring method is used instead.)

    Returns:
    (np.array) evaluation scores and statistics about the population.
    Scores have the same order as the population. The higher the value, the fitter the individuum.

    Parameters:
    population      -   (np.array)  the population to evaluate
    inputs          -   (np.array)  samples of inputs that are used to evaluate the individuals (WANNs) performances
    targets         -   (np.array)  samples of targets for evaluation; target values for the individuals (WANNs) predictions
    weight_values   -   (list)      a list of weight values that will be single shared over the WANNs
    rank_prob       -   [0,1]       probability that ranking is performed via number of connections and mean loss
                                    (instead of via mean loss and min loss)
    tau             -   [0,1]       scoring parameter: balances between number of connections and mean loss
    phi             -   [0,1]       scoring parameter: balances between min loss and mean loss       
    """
    
    # stats for complete population
    performances = []      # loss
    n_layers = []          # number of layers in WANN
    n_connections = []     # number of enabled connections

    weight_values = np.array(weight_values)

    # evaluate each individuum
    for indiv in tqdm(population):

        # performance per weight of one individual
        performance = evaluate_individuum(indiv, weight_values, inputs, targets, **hyper)
        # enabled connections and number of layers of individuum
        n_cons, n_lays = indiv.get_complexity()

        n_connections.append(n_cons)
        n_layers.append(n_lays)
        performances.append(performance)

    # make np.arrays from stats
    performances = np.array(performances)
    n_cons = np.array(n_connections)
    n_layers = np.array(n_layers)
    
    # get mean losses, best losses and best weights for best losses
    mean_losses = performances.mean(axis=1)
    min_losses = performances.min(axis=1)
    if len(weight_values) > 0: 
        # best weights for best losses
        best_weights = weight_values[np.argmin(performances, axis=1)] 
    else:
        best_weights = 0

    # normalizations
    normed_mean_losses = mean_losses / mean_losses.max()

    # scoring
    # in 0.8 of the cases ranking is done via mean performance and number of connections
    if np.random.random() <= rank_prob:
        # 0 - best score / 1 - worst score (for tau == 0.5)
        normed_n_cons = n_cons / n_cons.max()
        # naive approach instead of pareto dominance ranking
        scores = tau * normed_n_cons + (1 - tau) * normed_mean_losses
    else:
    # in 0.2 of the cases ranking is done via
    # mean performance and max performance
        # 0 - best score / 1 - worst score (for phi == 0.5)
        normed_min_losses = min_losses / min_losses.max()
        # naive approach instead of pareto dominance ranking
        scores = phi * normed_min_losses + (1 - phi) * normed_mean_losses

    # inverte the scores to get fitness scores: the higher the better! (maximization task)
    eval_scores = 1 - scores

    gen_statistics = {
          "mean_losses": mean_losses, 
          "min_losses": min_losses, 
          "best_weights": best_weights,
          "n_layers": n_layers,
          "n_cons": n_cons
    }

    return eval_scores, gen_statistics


def evaluate_individuum(individuum, weight_values, inputs, targets, n_rollouts=5, weight_type="shared", **kwargs):
    """ Measure performance of a single indviduum. 
    
    Returns:
    (np.array) Single performance values (the mean loss) for each weight, averaged over all rollouts.

    Parameters:
    individuum      -   (Individuum) WANN of which the performance should be measured
    weight_values   -   (list)       weight values that should each be used as single shared weight in the WANN
    inputs          -   (np.array)   inputs for the WANN (WANN should predict on them)
    targets         -   (np.array)   targets for the WANN (compare WANN prediction with them)
    n_rollouts      -   (int)        how often the WANNs performance per weight should be measured
    """

    # indicator for random weights
    if weight_type == "shared":
        assert len(weight_values) > 0, "No weight values for shared weight given!"
    elif weight_type == "random":
        weights_shape = individuum.get_genome().shape
        weight_values = [np.random.random(weights_shape) - 0.5]
    else:
        raise Exception("Invalid value for weight_type")

    performance_scores = np.empty((n_rollouts, len(weight_values)))
  
    for rollout in range(n_rollouts):

        # get input and target samples for this rollout
        input_samples = inputs[rollout]
        target_samples = targets[rollout]

        for weight_ix, weight in enumerate(weight_values):

            # get the ouput for this input and weight from the inviduum
            outputs = individuum.predict(input_samples, weight)
            logits = softmax(outputs, axis=1)
            losses = log_loss(target_samples, logits)

            # save the loss according to current weight and rollout
            performance_scores[rollout, weight_ix] = np.mean(losses)

    # return loss for each weight (averaged over all rollouts)
    return np.mean(performance_scores, axis=0)
    

def sample_data(X, y, n_rollouts):
    """ Samples Data from the task's data set.

    Parameters:
    n_rollouts  -  (int) number of repetitions for each weight value in evaluation.
    """
    inputs = X[np.random.randint(0, X.shape[0], size=(n_rollouts, 1000)), :]
    targets = y[np.random.randint(0, X.shape[0], size=(n_rollouts, 1000))]   

    return inputs, targets


        
if __name__ == "__main__":

    main(**hyper)

