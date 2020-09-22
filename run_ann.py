""" Script to run fully-connected comparable ANN experiments. """

from src.utilities import TF_LOSS_FUNCTIONS, EVAL_FUNCTIONS, load_dataset, plot_stats, titlelize

from tqdm import tqdm

import tensorflow.keras.layers as kl
import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Performs training of a ANN.")

parser.add_argument("dataset_name", type=str, help="Name of dataset: One of 'mnist', 'forestfires'.")

parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for ANN.")
parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs to train ANN.")


class ANN(kl.Layer):
    """ ANN Fully-Connected Model Definition. """
    
    def __init__(self, n_inputs=784, n_outputs=10, **kwargs):

        super().__init__()

        self.hidden_layer = kl.Dense(units=n_outputs)

        self.n_hidden_layers = 1
        self.n_connections = n_inputs * n_outputs

        
    def call(self, x):

        x = self.hidden_layer(x)

        return x
        


def main(dataset_name="mnist", learning_rate=0.05, batch_size=50, n_epochs=3, loss_name="cce", eval_name="acc", **hyper):
    """ Main loop of ANN training. """

    print("Loading train data ...")
    X_train, y_train = load_dataset(dataset_name, split="train")

    hyper["n_inputs"], hyper["n_outputs"] = X_train.shape[1], y_train.shape[1]
    total = X_train.shape[0] // batch_size

    train_dataset = make_tf_dataset(X_train, y_train)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    print("Loading test data ...")
    X_test, y_test = load_dataset(dataset_name, split="test")
    test_dataset = make_tf_dataset(X_test, y_test)
    test_dataset = test_dataset.batch(batch_size)

    print("Creating ANN ...")
    model = ANN(**hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_func = TF_LOSS_FUNCTIONS[loss_name]
    eval_func = EVAL_FUNCTIONS[eval_name]

    losses = []
    eval_scores = []
    n_connections = []

    tf.keras.backend.clear_session()

    print("Training ANN ...")

    for epoch in range(n_epochs):

        print("Epoch", epoch)

        epoch_losses, epoch_eval_scores = run_epoch(model, optimizer, train_dataset, test_dataset, loss_func, eval_func, dataset_name, total)

        losses.extend(epoch_losses)
        eval_scores.extend(epoch_eval_scores)

    n_hidden_layers = np.ones(total * n_epochs) * model.n_hidden_layers
    n_connections = np.ones(total * n_epochs) * model.n_connections

    losses = np.array(losses)
    eval_scores = np.array(eval_scores)

    title = "{} Epochs of ANN Training on {} \nwith Learning Rate {} and Batch Size {}".format(n_epochs, titlelize(dataset_name), learning_rate, batch_size)
    plot_fp = "plots/ann_{}.png".format(dataset_name)

    if not os.path.exists("plots"):
        os.mkdir("plots")

    if dataset_name == "mnist":
        sample_limit = 30_000
    elif dataset_name == "forestfires":
        sample_limit = 20_000
    else:
        raise Exception("Invalid datset name '{}'!".format(dataset_name))
    
    plot_stats(losses, n_connections, n_hidden_layers, eval_scores, eval_name, loss_name, batch_size, title, plot_fp, sample_limit)
    


def run_epoch(model, optimizer, train_dataset, test_dataset, loss_func, eval_func, dataset_name, total):
    """ Runs an epoch of ANN training and testing. """
    
    losses = []
    mean_eval_scores = []

    for x, y in tqdm(train_dataset, total=total):

        with tf.GradientTape() as tape:

            y_pred = model(x)
            loss = loss_func(y, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss)

        eval_scores = []

        for x, y in test_dataset:
            
            outputs = model(x).numpy()

            if dataset_name == "mnist":
                y_pred = np.argmax(outputs, axis=1)
                y = np.argmax(y, axis=1)
            elif dataset_name == "forestfires":
                y_pred = outputs[:, 0] * 300

            eval_score = eval_func(y, y_pred)
            eval_scores.append(eval_score)

        mean_eval_scores.append(np.mean(eval_scores))

    return losses, mean_eval_scores


def make_tf_dataset(X, y):
    """ Converts a numpy dataset to a tensorflow dataset. """

    X_t = tf.convert_to_tensor(X)
    y_t = tf.convert_to_tensor(y)

    return tf.data.Dataset.from_tensor_slices((X_t, y_t))


if __name__ == "__main__":
        
    args = parser.parse_args()
    params = vars(args)

    if args.dataset_name == "mnist":

        params["loss_name"] = "cce"
        params["eval_name"] = "acc"
        params["batch_size"] = 100

    elif params["dataset_name"] == "forestfires":

        params["loss_name"] = "mse"
        params["eval_name"] = "mae"
        params["batch_size"] = 10
    
    else:

        raise Exception("Invalid datset name '{}'!".format(params["dataset_name"]))

    print("\nPARAMS:\n", params, "\n")
            
    main(**params)
