# computes accuracies on test mnist dataset for stored best individuums and stores results in log/test/
from utilities import load_dataset, get_experiment_name
from individuum import Individuum

from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

# TODO check hyper parameters
# TODO more clear comments


# constants
SHARED_WEIGHT_VALUES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

EVAL_FUNCTIONS = {
    "acc": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
}

HYPER = {
    "n_gen": 3,             # number of generations the WANN-algorithm should (!) have
    "pop_size": 64,         # size of population
    "weight_values": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], # single shared weight values of the network
    "weight_type": "shared",# weights can either be shared or random
    "dataset_name": "mnist",# type of dataset
    "init_activation": 1,   # ReLU, MNIST specific
    "ratio_enabled": 0.05,   # probability of connection being enabled when individuum is initialized
    "eval_name": "acc"      # acc for accuracy or mae for mean absolute error
}

def test(hyper):
    """ Tests the best individuum per generation. Writes the results out into log/test/
    """
    # save mean evaluation scores
    mean_eval_scores = []
    
    # get experiment name
    exp_name = get_experiment_name(**hyper)     
    
    # get experiment folder path
    exp_folder = "best_individuums/" + exp_name
    
    # load test data set
    X, y = load_dataset(hyper["dataset_name"], split="test")
    
    # de-one-hot-encode
    y_true = np.argmax(y, axis=1)
    
    hyper["n_inputs"], hyper["n_outputs"] = X.shape[1], y.shape[1]

    # get eval function
    eval_func = EVAL_FUNCTIONS[hyper["eval_name"]]

    for gen in tqdm(range(0, hyper["n_gen"])):
        
        indiv_file_name = "_".join(["best", "gen", str(gen)])

        indiv_path = exp_folder + "/" + indiv_file_name

        # check if an individuum exists in this generation (algorithm might have stopped because not enough memory could be allocated)
        if os.path.isfile(indiv_path + "_layers"):
                        
            # create Individuum
            indiv = Individuum(**hyper)
            indiv.load_from(indiv_path)
            
            weight_values = np.array(SHARED_WEIGHT_VALUES, dtype=np.float64) 
            eval_scores = []            
            
            # predict and evaluate WANN
            for weight in weight_values:
                if hyper["weight_type"] == "random": 
                    weight = np.random.random(indiv.get_genome().shape) - 0.5

                # predict with single shared weight
                outputs = indiv.predict(X, weight)

                # decode
                y_pred = np.argmax(outputs, axis=1)

                eval_score = eval_func(y_true, y_pred)
                
                eval_scores.append(eval_score)
            
            # get mean of eval scores of one WANN (different weights)
            mean_eval_scores.append(np.mean(eval_scores))
                
        else: # in case file not found
            print("Individuum for", gen, ". generation does not exist")

        
    # save results
    log_folder = "log/test/" + exp_name + "/"
    os.mkdir(log_folder)
    pd.DataFrame({hyper["eval_name"]: mean_eval_scores}).to_csv(log_folder + "eval_scores.csv")

if __name__ == "__main__":
    test(HYPER)
    
