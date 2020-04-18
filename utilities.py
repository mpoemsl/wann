import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2

def get_experiment_name(dataset_name="mnist", n_gen=128, pop_size=64, weight_type="random", prob_crossover=0.0, **kwargs):
    """ Returns name for experiment given a unique subset of hyperparameters. """


    print("utilities", dataset_name)
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


