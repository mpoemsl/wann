# computes accuracies on test mnist dataset for stored best individuums and stores results in log/test/
from utilities import load_dataset
from individuum import Individuum

import numpy as np
import os

# TODO check hyper parameters
# TODO more clear comments

# initialization of hyper parameters
hyper = {
    "n_gen": 3,             # number of generations the WANN-algorithm should (!) have
    "pop_size": 64,         # size of population
    "weight_values": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], # single shared weight values of the network
    "weight_type": "random",# weights can either be shared or random
    "dataset_name": "mnist",# type of dataset
    "init_activation": 1,   # ReLU, MNIST specific
    "ratio_enabled": 0.05   # probability of connection being enabled when individuum is initialized
}

def test(n_gen=3, pop_size=64, weight_type="random", dataset_name="mnist", **hyper):
    """ Tests the best individuum per generation. Writes the results out into log/test/
    """
    exp_file_name = "best_individuums/" + "_".join(["experiment", str(3), str(64), weight_type])
    
    # load test data set
    X, y = load_dataset(dataset_name, split="test")
    hyper["n_inputs"], hyper["n_outputs"] = X.shape[1], y.shape[1]

    for gen in range(1, (n_gen+1)):
        
        indiv_file_name = "_".join(["best", "gen", str(gen)])
        indiv_path = exp_file_name + "/" + indiv_file_name

        # check if an individuum exists in this generation (algorithm might have stopped because not enough memory could be allocated)
        if os.path.isfile(indiv_path + "_layers"):
                        
            # create Individuum
            indiv = Individuum(**hyper)
            indiv.load_from(indiv_path)
            
            # for random weights: predict 10 times to get more averaged results
            if (weight_type == "random"):
                for predict_round in range(10):
                    # predict with random weights
                    print(predict_round)
            else: # for shared weights: predict for each weight and average
                for weight in weight_values:
                    # predict with single shared weight
                    print(weight)
                

        else:
            print("Individuum for", gen, ". generation does not exist")

        #try:
        #    with open(indiv_path, "rb") as f:
        #        print("found file")
        #except IOError:
        #    print("File for wanted individuum does not exist.")


test()
    
