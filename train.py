from genetic_algorithm import evolve_population
from individuum import Individuum

from sklearn.metrics import log_loss
from scipy.special import softmax
from copy import deepcopy
from tqdm import tqdm

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
import os

# initialization of hyper parameters
hyper = {
    "n_gen": 3,         # number of generations: they do 4096
    "pop_size": 64,       # size of population: they do 960
    "init_activation": 1,  # ReLU, MNIST specific
    "ratio_enabled": 0.05, # probability of connection being enabled when individuum is initialized
    "cull_ratio": 0.2,     # percentage of unfittest individuals who get excluded from breeding
    "elite_ratio": 0.2,    # percentage of fittest individuals who pass on to the new population unchanged
    "autogamy": 1.0,       # percentage of how often no crossover takes place, but the best genome is passed on after mutating
    "n_cross_points": 4,   # number of crossing points in crossover
    "prob_add_node": 0.25,  # probability of adding a node as mutation
    "prob_add_con": 0.25,   # probability of adding a connection as mutation
    "prob_change_activation": 0.5, # probability of changing the activation function as mutation
    "weight_values": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
    "weight_type": "random", # shared or random
    "n_rollouts": 1,  
    "rank_prob": 0.8,
    "tournament_size": 32,
    "tau": 0.5,
    "phi": 0.5,    
}

def main(n_gen=5, **hyper):
    """ Main Loop for the WANN construction.
    """

    hyper["experiment_name"] = "experiment_{}_{}_{}".format(n_gen, hyper["pop_size"], hyper["weight_type"])

    print("Creating folders for experiment {} ...".format(hyper["experiment_name"]))
    os.mkdir("best_individuums/" + hyper["experiment_name"])
    os.mkdir("log/train/" + hyper["experiment_name"])

    print("Loading MNIST training data ...")
    X, y = load_mnist()
    hyper["n_inputs"], hyper["n_outputs"] = X.shape[1], y.shape[1]

    print("Initializing Population ...")
    population = init_population(**hyper)   # initialize the population

    for gen in range(n_gen):

        print("\nGeneration", gen + 1)
        start = time.time()

        print("Sampling data ...")
        inputs, targets = sample_data(X, y, hyper["n_rollouts"])

        print("Evaluating population ... ")
        eval_scores, gen_statistics = evaluate_population(population, inputs, targets, **hyper)  # evaluate the performance of population

        print("Evolving population ...")
        population = evolve_population(population, eval_scores, gen=gen, **hyper)  # create new population based on evaluation

        print("Generation lasted {:4f} seconds.".format(time.time() - start))

        # Save statistics in log
        pd.DataFrame(gen_statistics).to_csv("log/train/{}/stats_gen_{}".format(hyper["experiment_name"], gen))

    print("Finished running {} generations.".format(n_gen))


def init_population(pop_size=20, **hyper):
    """ Initializes the population by creating a lot of new individuals.
    """
    population = []
    for _ in range(pop_size):
        indiv = Individuum(**hyper)
        population.append(indiv)

    return np.array(population)


def evaluate_population(population, inputs, targets, weight_values=[], rank_prob=0.8, tau=0.5, phi=0.5, **hyper):
    """ Evaluates a complete population. Returns a numpy array of evaluation scores.
    Scores have the same order as the population array.
    The higher the value, the fitter the individuum.
    """

    performances = []
    n_layers = []
    n_connections = []     # number of enabled connections

    weight_values = np.array(weight_values)

    # evaluate each individuum
    for indiv in tqdm(population):

        # performance per weight of one individual
        performance = evaluate_individuum(indiv, weight_values, inputs, targets, **hyper)
        n_cons, n_lays = indiv.get_complexity()

        n_connections.append(n_cons)
        n_layers.append(n_lays)
        performances.append(performance)

    performances = np.array(performances)
    n_cons = np.array(n_connections)
    n_layers = np.array(n_layers)

    mean_losses = performances.mean(axis=1) # mean losses
    min_losses = performances.min(axis=1) # best losses

    if len(weight_values) > 0:
        best_weights = weight_values[np.argmin(performances, axis=1)] # best weights for best losses
    else:
        best_weights = 0

    # normalizations
    normed_mean_losses = mean_losses / mean_losses.max()

    # scoring
    # paper: in 0.8 of the cases ranking is done via
    # mean performance and number of connections
    if np.random.random() <= rank_prob:
        # 0 - best score. 1 - worst score. (for tau == 0.5)
        normed_n_cons = n_cons / n_cons.max()
        scores = tau * normed_n_cons + (1 - tau) * normed_mean_losses
    else:
    # in 0.2 of the cases ranking is done via
    # mean performance and max performance
        # 0 - best score. 1 - worst score. (for phi == 0.5)
        normed_min_losses = min_losses / min_losses.max()
        scores = phi * normed_min_losses + (1 - phi) * normed_mean_losses

    """ Naive ranking method. Ranks a complete population according to mean loss
    and either number of connections XOR minimum loss performance. 
    The parameter tau balances mean loss and connections.
    The parameter phi balances mean loss and minimal loss.
    Returns a numpy array with a ranking, ordered in the same manner like the mean loss values.
    """

    eval_scores = 1 - scores

    gen_statistics = {
          "mean_losses": mean_losses, 
          "min_losses": min_losses, 
          "best_weights": best_weights,
          "n_layers": n_layers,
          "n_cons": n_cons
    }

    # maximize
    return eval_scores, gen_statistics


def evaluate_individuum(individuum, weight_values, inputs, targets, n_rollouts=5, weight_type="shared", **kwargs):
    """ Measure performance of a single indviduum. Returns the single performance values for each weight, averaged over all rollouts.
    individuum must stem from the class Individuum().
    inputs are the input samples that are fed into the WANN. Must be coherent with targets and num_rollouts.
    targets are the labels for the inputs. Must be coherent with inputs and num_rollouts.
    n_rollouts indicates how many times the WANNs performance for the weights should be measured.
    weight_values carries the weights that are each used as one shared weight in the WANN.
    Returns a numpy array with the performance results for each rollout and weight.
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

    return np.mean(performance_scores, axis=0) # loss for each weight (averages over all rollouts)
    

def sample_data(X, y, n_rollouts):
    """ Samples from training data. """

    inputs = X[np.random.randint(0, X.shape[0], size=(n_rollouts, 1000)), :]
    targets = y[np.random.randint(0, X.shape[0], size=(n_rollouts, 1000))]   

    return inputs, targets


def load_mnist():
    """ Loads and preprocessed MNIST training data. """

    train_dataset = tfds.load(name="mnist", split="train")

    train_images = np.array([sample["image"] for sample in tfds.as_numpy(train_dataset)])
    train_labels = np.array([sample["label"] for sample in tfds.as_numpy(train_dataset)])

    processed_train_images = np.array([downsize_and_deskew(img / 255.0) for img in train_images])
    processed_train_images = processed_train_images.reshape(train_images.shape[0], -1)

    processed_train_labels = np.zeros((train_labels.shape[0], 10), dtype=np.float64)
    processed_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1.0

    return processed_train_images, processed_train_labels


def downsize_and_deskew(img, tgt_shape=(16, 16)):
    """ MNIST preprocessing adopted from WANN code base at https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANN. """
    
    downsized_img = cv2.resize(img, tgt_shape)
    moments = cv2.moments(downsized_img)
    
    if abs(moments["mu02"]) < 1e-2:
        return downsized_img
    else:
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, -0.5 * tgt_shape[0] * skew], [0, 1, 0]])
        return cv2.warpAffine(downsized_img, M, tgt_shape, flags=(cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR))  

        
if __name__ == "__main__":

    main(**hyper)

