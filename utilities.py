""" Utility functions and constants for running WANN experiments. """

from sklearn.metrics import log_loss, mean_squared_error, accuracy_score, mean_absolute_error
from scipy.special import softmax

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


# adopted from Gaier and Ha, 2019
SHARED_WEIGHT_VALUES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]


TF_LOSS_FUNCTIONS = {
    "cce": tf.keras.losses.CategoricalCrossentropy(),
    "mse": tf.keras.losses.MeanSquaredError()
}

LOSS_FUNCTIONS = {
    "cce": lambda y_true, y_pred: log_loss(y_true, softmax(y_pred, axis=1)),
    "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
}


EVAL_FUNCTIONS = {
    "acc": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
}


ACTIVATION_DICT =  {
    1: lambda x: np.maximum(0, x),          # relu
    2: lambda x: x,                         # linear
    3: lambda x: 1.0 * (x > 0.0),           # unsigned step function
    4: lambda x: np.sin(np.pi * x),         # sin
    5: lambda x: np.exp(-(x * x) / 2.0),   # gausian with mean 0 and sigma 1
    6: lambda x: np.tanh(x),                # hyperbolic tangent (tanh) signed
    7: lambda x: (np.tanh(x / 2.0) + 1.0) / 2.0, # sigmoid unsigned ( 1 / (1 + exp(-x)) )
    8: lambda x: -x,                        # inverse
    9: lambda x: abs(x),                    # absolute value
    10: lambda x: np.cos(np.pi * x),        # cosine
    11: lambda x: x ** 2                    # squared
}


def get_experiment_name(dataset_name="mnist", n_gen=128, pop_size=64, weight_type="random", prob_crossover=0.0, **kwargs):
    """ Returns name for experiment given a unique subset of hyperparameters. """

    pc_str = "-".join(str(prob_crossover).split("."))
    return "experiment_{}_{}_{}_{}_{}".format(dataset_name, n_gen, pop_size, weight_type, pc_str)    


def load_dataset(dataset_name, split="train"):
    """ Loads and preprocesses MNIST training data. """

    if dataset_name == "mnist":

        dataset = tfds.load(name="mnist", split=split)

        images = np.array([sample["image"] for sample in tfds.as_numpy(dataset)])
        labels = np.array([sample["label"] for sample in tfds.as_numpy(dataset)])

        processed_images = np.array([downsize_and_deskew(img / 255.0) for img in images])
        processed_images = processed_images.reshape(images.shape[0], -1)
        
        # one-hot-encoding
        processed_labels = np.zeros((labels.shape[0], 10), dtype=np.float64)
        processed_labels[np.arange(labels.shape[0]), labels] = 1.0

        return processed_images, processed_labels

    elif dataset_name == "forestfires":
        
        # load dataset   
        if split == "train":

            dataset = tfds.load(name="forest_fires", split="train[20%:]")
            x = np.array([(sample["features"]["rain"], sample["features"]["wind"], sample["features"]["RH"], sample["features"]["temp"]) for sample in tfds.as_numpy(dataset)])
            y = np.array([sample["area"] for sample in tfds.as_numpy(dataset)])
            
            x_normalized = np.stack([(x[:, col] - x[:, col].min()) / (x[:, col].max() - x[:, col].min()) for col in range(x.shape[1])], axis=1)       

            # 300 is a upper limit for y
            y_normalized = y / 300
            y_normalized = y_normalized.reshape(-1, 1)            
            
            return x_normalized, y_normalized

        elif split == "test":

            dataset = tfds.load(name="forest_fires", split="train[:20%]")
            x = np.array([(sample["features"]["rain"], sample["features"]["wind"], sample["features"]["RH"], sample["features"]["temp"]) for sample in tfds.as_numpy(dataset)])
            y_raw = np.array([sample["area"] for sample in tfds.as_numpy(dataset)])
            
            x_normalized = np.stack([(x[:, col] - x[:, col].min()) / (x[:, col].max() - x[:, col].min()) for col in range(x.shape[1])], axis=1)       
            y_raw = y_raw.reshape(-1, 1)

            # y_pred must be: y_pred * 300
            return x_normalized, y_raw           

        else:
            raise Exception("Invalid value for parameter split")
        
    else:
        raise Exception("Invalid dataset name!")
    

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


def titlelize(word):
    """ Title-cases a word. """

    chars = list(word)
    chars[0] = chars[0].upper()

    return "".join(chars)


def plot_stats(mean_losses, mean_n_cons, mean_n_layers, mean_eval_scores, eval_name, loss_name, samples_per_step, title, plot_fp, sample_limit):
    """ Creates and saves illustrations of a WANN or ANN experiment. """

    n_samples = np.arange(mean_losses.shape[0]) * samples_per_step
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    axes[0, 0].plot(n_samples, mean_eval_scores, color="blue")
    axes[0, 0].set_ylabel(eval_name.upper())
    axes[0, 0].set_xlim(0, sample_limit)

    if eval_name == "acc":
        axes[0, 0].set_ylim(0, 1)

    axes[0, 0].set_title("Mean {} Score".format(eval_name.upper()))

    axes[0, 1].plot(n_samples, mean_losses, color="green")
    axes[0, 1].set_ylabel(loss_name.upper())
    axes[0, 1].set_xlim(0, sample_limit)
    axes[0, 1].set_title("Mean {} Loss".format(loss_name.upper()))

    axes[1, 0].plot(n_samples, mean_n_cons, color="red")
    axes[1, 0].set_xlabel("Number of Observed Samples")
    axes[1, 0].set_ylabel("Connections")
    axes[1, 0].set_xlim(0, sample_limit)
    axes[1, 0].set_xticks([0, sample_limit])
    axes[1, 0].set_title("Mean Number of Connections")

    axes[1, 1].plot(n_samples, mean_n_layers, color="yellow")
    axes[1, 1].set_xlabel("Number of Observed Samples")
    axes[1, 1].set_ylabel("Layers")
    axes[1, 1].set_xlim(0, sample_limit)
    axes[1, 1].set_ylim(0, 5)
    axes[1, 1].set_xticks([0, sample_limit])
    axes[1, 1].set_title("Mean Number of Hidden Layers")
    
    plt.suptitle(title)   

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.85, bottom=0.10)

    plt.savefig(plot_fp)
    plt.show()


