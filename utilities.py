import tensorflow_datasets as tfds
import numpy as np
import cv2

def load_dataset(dataset_name, split="train"):
    """ Loads and preprocesses MNIST training data. """

    if dataset_name == "mnist":

        dataset = tfds.load(name="mnist", split=split)

        images = np.array([sample["image"] for sample in tfds.as_numpy(dataset)])
        labels = np.array([sample["label"] for sample in tfds.as_numpy(dataset)])

        processed_images = np.array([downsize_and_deskew(img / 255.0) for img in images])
        processed_images = processed_images.reshape(images.shape[0], -1)

        processed_labels = np.zeros((labels.shape[0], 10), dtype=np.float64)
        processed_labels[np.arange(labels.shape[0]), labels] = 1.0

        return processed_images, processed_labels

    elif dataset_name == "forest_fires":
        pass
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
