# computes accuracies on test mnist dataset for stored best individuums and stores results in log/test/
from utilities import load_dataset, get_experiment_name
from individuum import Individuum

from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import os


# constants
SHARED_WEIGHT_VALUES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

EVAL_FUNCTIONS = {
    "acc": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
}

parser = argparse.ArgumentParser(description="Performs test evaluations on best individuals of a WANN experiment.")

parser.add_argument("train_experiment_folder", type=str, help="Path to a train experiment folder.")
parser.add_argument("eval_name", type=str, help="Evaluation function name (acc or mae)")


def main():
    """ Tests the best individuum per generation. Writes the results out into log/test/.
    """

    args = parser.parse_args()
    exp_name = args.train_experiment_folder.split("/")[-2]

    dataset_name, n_gen, pop_size, weight_type = exp_name.split("_")[1:]
    n_gen, pop_size = int(n_gen), int(pop_size)

    # save mean evaluation scores
    mean_eval_scores = []   
    
    # get experiment folder path
    exp_folder = args.train_experiment_folder
    eval_name = args.eval_name
    
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
        
        indiv_file_name = "_".join(["best", "gen", str(gen)])

        indiv_path = exp_folder + indiv_file_name

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
                    # decode
                    y_pred = np.argmax(outputs, axis=1)

                elif dataset_name == "forestfires":
                    # delog
                    y_pred = outputs * 300

                print(y_true, y_pred)
                eval_score = eval_func(y_true, y_pred[:, 0])
                
                eval_scores.append(eval_score)
            
            # get mean of eval scores of one WANN (different weights)
            mean_eval_scores.append(np.mean(eval_scores))
                
        else: # in case file not found
            print("Individuum for", gen, ". generation does not exist")
        
    # save results
    log_folder = "log/test/" + exp_name + "/"
    os.mkdir(log_folder)
    pd.DataFrame({eval_name: mean_eval_scores}).to_csv(log_folder + "eval_scores.csv")

if __name__ == "__main__":
    main()
    
