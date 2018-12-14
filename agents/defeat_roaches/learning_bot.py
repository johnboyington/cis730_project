from pysc2.agents import base_agent
from pysc2.lib import actions
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from transform_action import transform_action


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.setup_model()
        self.train_model()

    def setup_model(self):
        """Build ANN."""
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(750, activation='relu'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return

    def process_data(self, filename):
        """A utility used to convert logged data into a form accepted by ANN software."""

        # load the data and begin preprocessing
        storage = np.load(filename)
        np.random.shuffle(storage)
        storage = abs(storage)

        # split into observation data and the action that was taken
        data = storage[:, :-1]
        data[:, :961] = data[:, :961] / np.max(data[:, :961])
        labels = storage[:, -1]
        labels = labels

        return data, labels

    def train_model(self):
        """Train the ANN."""
        # grab data
        data, labels = self.process_data('data.npy')

        # fit the data to the model
        self.model.fit(data, labels, epochs=3, batch_size=64)

    def predict_action(self, obs):
        """Returns an action number given an observation."""
        row = []
        fs = obs.observation.feature_screen

        for map_type, screen in [('hp', fs.unit_hit_points), ('pid' ,fs.unit_type)]:
                screen = np.array(screen)
                # represent data as 31x31
                sizex = 31
                sizey = 31
                px = int(screen.shape[0] / sizex)
                py = int(screen.shape[1] / sizey)
                
                img = np.empty((sizex, sizey))
                
                # loop over data
                for i in range(sizex):
                    for j in range(sizey):
                        x = i*px
                        y = j*py
                        if map_type == 'hp':
                            img[i, j] = np.average(screen[x:x+px,y:y+py])
                        elif map_type == 'pid':
                            val = np.max(screen[x:x+px,y:y+py])
                            if val == 110:
                                val = 1
                            else:
                                val = 0
                            img[i, j] = val
                row += list(img.flatten())
        
        data = np.array(row).reshape(-1, len(row))
        data[:, :961] = data[:, :961] / np.max(data[:, :961])
        action = int(np.argmax(self.model.predict(data)))
        return action

    def can_do(self, obs, action):
        """Checks if an action is in the list of available actions."""
        return action in obs.observation.available_actions

    def step(self, obs, dummy):
        """This agent will either attack or move to a random square."""
        super(TerranAgent, self).step(obs)

        # first, pick a random number to represent some action from the action space
        action_id = self.predict_action(obs)

        # transform the action id into a target
        target = transform_action(action_id)

        if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
            return actions.FUNCTIONS.Attack_screen("now", target)

        return actions.FUNCTIONS.no_op()
