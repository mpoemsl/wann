from utilities import load_dataset, get_experiment_name
from genetic_algorithm import evolve_population
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

SHARED_WEIGHT_VALUES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

LOSS_FUNCTIONS = {
    "cce": lambda targets, outputs: log_loss(targets, softmax(outputs, axis=1)),
    "mse": lambda targets, outputs: np.square(targets - outputs).mean()
}

# initialization of hyper parameters
hyper = {
    "dataset_name": "mnist",        # name of dataset
    "loss_name": "cce",             # cross-entropy loss
    "n_gen": 3,                     # number of generations: paper does 4096
    "pop_size": 64,                 # size of population: paper does 960
    "sample_size": 1000,            # size of random sample that individuums are evaluated on
    "weight_type": "shared",        # weights can either be shared or random
    "ratio_enabled": 0.05,          # probability of connection being enabled when individuum is initialized
    "tau": 0.5,                     # parameter that balances off the ranking between number of connection and mean loss
    "phi": 0.5,                     # parameter that balances off the ranking between min loss and mean loss
    "cull_ratio": 0.2,              # percentage of unfittest individuals who get excluded from breeding
    "elite_ratio": 0.2,             # percentage of fittest individuals who pass on to the new population unchanged
    "tournament_size": 32,          # number of individuals competing to become parent of new kid
    "prob_crossover": 0.0,          # percentage of how often no crossover takes place, but the best genome is passed on after mutating
    "prob_add_node": 0.25,          # probability of adding a node as mutation
    "prob_add_con": 0.25,           # probability of adding a connection as mutation
    "prob_change_activation": 0.5,  # probability of changing the activation function as mutation
    "prob_rank_n_cons": 0.8         # probability that ranking is performed via number of connections and mean loss (instead of via mean loss and min loss)
}


def main(n_gen=5, dataset_name="mnist", **hyper):
    """ Main loop for the WANNs construction.
    Creates a population of WANNs and evolves it over several generations.

    Parameters:
    n_gen        -  (int) number of generations or how often the evolution should take place
    dataset_name -  (string) name of the dataset for which the WANN is constructed 
    """

    hyper["experiment_name"] = get_experiment_name(**hyper)

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
        inputs, targets = sample_data(X, y, **hyper)

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


def evaluate_population(population, inputs, targets, prob_rank_n_cons=0.8, tau=0.5, phi=0.5, weight_type="shared", loss_name="cce", **hyper):
    """ Evaluates a complete population via mean loss and either number of connections XOR minimum loss performance.
    (Attention! Paper uses pareto dominance ranking, this is not done here! A simple scoring method is used instead.)

    Returns:
    (np.array) evaluation scores and statistics about the population.
    Scores have the same order as the population. The higher the value, the fitter the individuum.

    Parameters:
    population      -   (np.array)  the population to evaluate
    inputs          -   (np.array)  samples of inputs that are used to evaluate the individuals (WANNs) performances
    targets         -   (np.array)  samples of targets for evaluation; target values for the individuals (WANNs) predictions
    rank_prob       -   [0,1]       probability that ranking is performed via number of connections and mean loss
                                    (instead of via mean loss and min loss)
    tau             -   [0,1]       scoring parameter: balances between number of connections and mean loss
    phi             -   [0,1]       scoring parameter: balances between min loss and mean loss       
    """
    
    # stats for complete population
    n_layers = np.empty(population.shape[0])                                # number of layers in WANN
    n_cons = np.empty(population.shape[0])                                  # number of enabled connections

    losses = []
    loss_func = LOSS_FUNCTIONS[loss_name]
    
    # evaluate each individuum
    for ix, individuum in enumerate(tqdm(population)):

        # gather weight values
        if weight_type == "shared":
            weight_values = np.array(SHARED_WEIGHT_VALUES, dtype=np.float64)
        elif weight_type == "random":
            weight_values = np.expand_dims(np.random.random(individuum.get_genome().shape) - 0.5, axis=0)

        # performance per weight of one individual
        losses.append(evaluate_individuum(individuum, weight_values, inputs, targets, loss_func, **hyper))

        # enabled connections and number of layers of individuum
        n_cons[ix], n_layers[ix] = individuum.get_complexity()
    
    # get mean losses, best losses and best weights for best losses
    losses = np.array(losses)
    mean_losses = losses.mean(axis=1)
    min_losses = losses.min(axis=1)

    # normalization
    normed_mean_losses = mean_losses / mean_losses.max()

    # scoring
    # in 0.8 of the cases ranking is done via mean performance and number of connections
    if np.random.random() <= prob_rank_n_cons:
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


    if weight_type == "random":
        best_weights = 0
    else:
        best_weights = weight_values[np.argmin(losses, axis=1)] 

    stats = {
          "mean_losses": mean_losses, 
          "min_losses": min_losses, 
          "best_weights": best_weights,
          "n_layers": n_layers,
          "n_cons": n_cons
    }

    # invert the scores to get fitness scores: the higher the better! (maximization task)
    return 1 - scores, stats


def evaluate_individuum(individuum, weight_values, inputs, targets, loss_func, **kwargs):
    """ Measure performance of a single indviduum. 
    
    Returns:
    (np.array) Single performance values (the mean loss) for each weight, averaged over all rollouts.

    Parameters:
    individuum      -   (Individuum) WANN of which the performance should be measured
    weight_values   -   (list)       weight values that should each be used as single shared weight in the WANN
    inputs          -   (np.array)   inputs for the WANN (WANN should predict on them)
    targets         -   (np.array)   targets for the WANN (compare WANN prediction with them)
    loss_func       -   (function)   loss function
    """


    losses = []
  
    for weight_ix, weight in enumerate(weight_values):

        # get the ouput for this input and weight from the inviduum
        outputs = individuum.predict(inputs, weight)
        losses.append(loss_func(targets, outputs))

    # return loss for each weight (averaged over all rollouts)
    return losses
    

def sample_data(X, y, sample_size=1000, **kwargs):
    """ Samples Data from the task's data set.

    Parameters:
    n_rollouts  -  (int) number of repetitions for each weight value in evaluation.
    """
    inputs = X[np.random.randint(0, X.shape[0], size=sample_size)]
    targets = y[np.random.randint(0, X.shape[0], size=sample_size)]   

    return inputs, targets


        
if __name__ == "__main__":

    main(**hyper)

