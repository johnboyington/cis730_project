from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.setup_model()
        self.train_model()

    def setup_model(self):
        """Build ANN."""
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(124, activation='relu'))
        self.model.add(layers.Dense(61*81*2, activation='softmax'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return

    def process_data(self, filename):
        """A utility used to convert logged data into a form accepted by ANN software."""

        # load the data
        storage = np.load(filename)

        # grab only data with a positive score
        good_rows = storage[storage[:, -2] > 0]

        # split into observation data and the action that was taken
        data = good_rows[:, :-2]
        labels = good_rows[:, -1]

        return data, labels

    def train_model(self):
        """Train the ANN."""
        # grab data
        data, labels = self.process_data('logged_data100.npy')

        # fit the data to the model
        self.model.fit(data, labels, epochs=1, batch_size=32)

    def predict_action(self, obs):
        """Returns an action number given an observation."""
        row = []
        fs = obs.observation.feature_screen

        for screen in [fs.unit_hit_points, fs.unit_type]:
            row += list(screen.flatten())

        data = np.array(row).reshape(-1, len(row))
        return np.argmax(self.model.predict(data))

    def can_do(self, obs, action):
        """Checks if an action is in the list of available actions."""
        return action in obs.observation.available_actions

    def step(self, obs):
        """This agent will either attack or move to a random square."""
        super(TerranAgent, self).step(obs)

        # establish some ground rules
        x_space_size = 61
        y_space_size = 81
        action_space_size = x_space_size * y_space_size * 2

        # first, pick a random number to represent some action from the action space
        rho = self.predict_action(obs)
        rho_stored = rho

        # determine if action is attack (<1200) or move (>=1200)
        if rho < (action_space_size / 2):
            action = 'attack'
        else:
            action = 'move'
            rho -= (action_space_size / 2)

        # calc x and y from the random number
        x = rho // x_space_size
        y = rho % x_space_size
        target = (x, y)

        if action == 'attack':
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", target), rho_stored
        elif action == 'move':
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", target), rho_stored

        return actions.FUNCTIONS.no_op(), rho_stored
