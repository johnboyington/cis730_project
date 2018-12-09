

class Data_Container(object):
    """A container for data used to train our ANN."""
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.discounts = []
        self.actions = []
    
    def log_step(self, timestep, action):
        """This method takes in a pysc2 timestep and an action index 
        and stores info from it."""
        self.observations.append(timestep.observation)
        self.rewards.append(timestep.reward)
        self.discounts.append(timestep.discount)
        self.actions.append(action)