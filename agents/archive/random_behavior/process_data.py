import numpy as np


def process_data(filename):
    """A utility used to convert logged data into a form accepted by ANN software."""

    # load the data
    storage = np.load(filename)

    # grab only data with a positive score
    good_rows = storage[storage[:, -2] > 0]

    # split into observation data and the action that was taken
    data = good_rows[:, :-2]
    labels = good_rows[:, -1]

    return data, labels


if __name__ == '__main__':
    process_data('logged_data100.npy')
