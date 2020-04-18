import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2

# TODO unlogged / logged data for fire datset?
# TODO shuffle datasets?

def get_experiment_name(dataset_name="mnist", n_gen=128, pop_size=64, weight_type="random", **kwargs):
    """ Returns name for experiment given a unique subset of hyperparameters. """

    return "experiment_{}_{}_{}_{}".format(dataset_name, n_gen, pop_size, weight_type)    
    

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

    elif dataset_name == "forest_fires":
        
        # load dataset   
        if split == "train":

            dataset = tfds.load(name="forest_fires", split="train[:80%]")
            rain = np.array([sample["features"]["rain"] for sample in tfd.as_numpy(dataset)])
            wind = np.array([sample["features"]["wind"] 
            targets = np.array([sample["area"] for sample in tfds.as_numpy(dataset)])

            print(features)
            
            processed_features = np.array([normalize(sample) for sample in features])
            processed_targets = np.array([make_log(sample) for sample in targets])
            
            print("finished preprocessing")            
            
            return processed_features, processed_targets

        elif split == "test":

            dataset = tfds.load(name="forest_fires", split="train[80%:]")
            inputs, targets_logged, targets_unlogged = dataset.map(normalize)
            
            return inputs, targets_unlogged            

        else:
            raise Exception("Invalid value for parameter split")
        
        
        
        
        return processed_dataset
        
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

def normalize(row):
    """ Forest Fires data preprocessing step in order to normalize the features.
    """

    temp = (row["features"]["temp"] - 2.2) / 31.1
    wind = (row["features"]["wind"] - 0.4) / 9.0
    rain = row["features"]["rain"] / 6.4
    humy = (row["features"]["RH"] - 15.0) / 85.0

    x = tf.stack([temp, wind, rain, humy])
    y_unlogged = row["area"]
    y_logged = tf.math.log(row["area"] + 1.0) / 10

    return x, y_logged, y_unlogged

def make_log(row):
    return log(row["area"] + 1.0) / 10

load_dataset("forest_fires", split="train")

