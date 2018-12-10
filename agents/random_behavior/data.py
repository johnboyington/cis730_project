import numpy as np


class Data_Container(object):
    """A container for data used to train our ANN."""
    def __init__(self):
        self.data = []
    
    def log_step(self, timestep, action):
        """This method takes in a pysc2 timestep and an action index 
        and stores info from it."""
        
        #obs = timestep.observation.feature_units
        obs = [-1, -2, -3]
        
        row = obs
        row.append(timestep.reward)
        row.append(action)
        self.data.append(np.array(row))
        return
    
    
    def save_data(self, filename):
        """A utility to store the data in a human readable numpy array."""
        self.data = np.array(self.data)
        np.savetxt(filename, self.data, delimiter=',')
                
        
        