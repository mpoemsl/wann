""" Script to compute evaluation scores on the test data of WANN experiments. """

from src.utilities import SHARED_WEIGHT_VALUES, EVAL_FUNCTIONS, load_dataset, get_experiment_name
from src.individuum import Individuum

from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Performs test evaluations on best individuums of a WANN experiment.")
parser.add_argument("exp_folder", type=str, help="Path to WANN experiment folder.")

def main(exp_folder):
    """ Tests the best individuum per generation. Writes the results out into log/test/.
    """

    exp_name = exp_folder.split("/")[-2]

    dataset_name, n_gen, pop_size, weight_type = exp_name.split("_")[1:-1]
    n_gen, pop_size = int(n_gen), int(pop_size)

    if dataset_name == "mnist":
        eval_name = "acc"
    elif dataset_name == "forestfires":
        eval_name = "mae"
    else:
        raise Exception("Invalid datset name '{}'!".format(dataset_name))

    print("Calculating {} for {} ...".format(eval_name.upper(), exp_name))

    # save mean evaluation scores
    mean_eval_scores = []   
    
    # load test data set
    X, y = load_dataset(dataset_name, split="test")
    
    # de-one-hot-encode
    y_true = np.argmax(y, axis=1)
    
    # minimal hyperparameters for Individuum
    hyper = {
        "n_inputs": X.shape[1],
        "n_outputs": y.shape[1]
    }

    # get eval function
    eval_func = EVAL_FUNCTIONS[eval_name]

    for gen in tqdm(range(0, n_gen)):
        
        indiv_file_name = "best_gen_" + str(gen)

        indiv_path = "experiments/{}/train/best_individuums/{}".format(exp_name, indiv_file_name)

        # check if an individuum exists in this generation (algorithm might have stopped because not enough memory could be allocated)
        if os.path.isfile(indiv_path + "_layers"):
                        
            # create Individuum
            indiv = Individuum(**hyper)
            indiv.load_from(indiv_path)
            
            weight_values = np.array(SHARED_WEIGHT_VALUES, dtype=np.float64) 
            eval_scores = []            
            
            # predict and evaluate WANN
            for weight in weight_values:
                if weight_type == "random": 
                    weight = np.random.random(indiv.get_genome().shape) - 0.5

                # predict with single shared weight
                outputs = indiv.predict(X, weight)
        
                if dataset_name == "mnist":
                    y_pred = np.argmax(outputs, axis=1)
                elif dataset_name == "forestfires":
                    y_pred = outputs[:, 0] * 300

                eval_score = eval_func(y_true, y_pred)
                
                eval_scores.append(eval_score)
            
            # get mean of eval scores of one WANN (different weights)
            mean_eval_scores.append(np.mean(eval_scores))
                
        else: # in case file not found
            print("Individuum for", gen, ". generation does not exist")
        
    # save results
    pd.DataFrame({eval_name: mean_eval_scores}).to_csv("experiments/{}/test/eval_scores.csv".format(exp_name))

if __name__ == "__main__":

    args = parser.parse_args()
    main(args.exp_folder)
    
