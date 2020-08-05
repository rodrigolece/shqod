import numpy as np
import math

"""
Utilities for SHQ path analysis
"""

def path_velocity(path):
    """
    No velocity consideration, assume uniform time

    :param path_in: 2D Numpy input with each row being the coordinates visited
    :return: The array of adjacent row-wise differences
    """
    return path[1:] - path[:-1]


def path_integrate(velocity):
    """
    No velocity consideration, assume uniform time

    :param path_in: 2D Numpy input with each row being the velocities at each time
    :return: The array of cumulative sums
    """
    return np.vstack((np.zeros((1, velocity.shape[1])), np.cumsum(velocity, axis=0)))


def gaussian_2d_filter(n, step_size):
    # Odd sized
    pre_output = abs(np.matmul(np.ones(2*n+1).reshape(-1, 1), np.arange(-n, n+1).reshape(1, -1)))
    pre_output += pre_output.transpose()
    pre_output = pre_output * pre_output
    pre_output *= (-step_size/2)
    output = np.exp(pre_output)/np.sqrt(2*math.pi)
    return output


def filter_pointcloud(sample, labels_in, return_detailed = False):
    """
    Filter a pointcloud by integer label
    Assume that sample is a (k x n) list where there are n dimensions, and depth is a shape (k,) ndarray encoding labels
    Return a list of (m x n) arrays where m is the number of samples that has each label type.
    """
    labels = list(labels_in)
    num_samples = sample.shape[0]
    assert len(labels) == num_samples, "the number of samples must be the same as the number of depth data"
    unique_labels = np.unique(labels)
    unique_labels_hor_ext = np.matmul(unique_labels.reshape(-1, 1), np.ones((1, num_samples)))
    all_labels_vert_ext = np.matmul(np.ones((len(unique_labels) , 1)), np.array(labels).reshape(1,-1))
    sample_binary_labels = np.equal(unique_labels_hor_ext, all_labels_vert_ext) # each column is a binary one-hot array indicating its label
    vanilla_indices = np.array(range(len(labels)))

    if not return_detailed:
        return [sample[sample_binary_labels[i]] for i in range(len(unique_labels))]
    else:
        return {"points": [sample[sample_binary_labels[i]] for i in range(len(unique_labels))],
                "indices": [vanilla_indices[sample_binary_labels[i]] for i in range(len(unique_labels))]}

