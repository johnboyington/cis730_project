import pickle


def process_data(filename):
    """A utility used to convert logged data into a form accepted by ANN software."""
    
    with open(filename, 'rb') as F:
        storage = pickle.load(F)
    
    
    obs = storage.observations[0]
    print(type(obs))
    print(obs['single_select'])
    


if __name__ == '__main__':
    process_data('logged_data.p')