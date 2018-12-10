import numpy as np


def process_data(filename):
    """A utility used to convert logged data into a form accepted by ANN software."""
    
    storage = np.load(filename)
    scores = storage[:, -2]
    
    
    print(len(scores[scores > 0]))


if __name__ == '__main__':
    process_data('logged_data100.npy')
