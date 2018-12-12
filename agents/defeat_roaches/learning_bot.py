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
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(121*101, activation='softmax'))
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return

    def process_data(self, filename):
        """A utility used to convert logged data into a form accepted by ANN software."""

        # load the data and begin preprocessing
        storage = np.load(filename)
        storage = abs(storage)

        # split into observation data and the action that was taken
        data = storage[:, :-2]
        data = data / np.max(data)
        labels = storage[:, -1]
        labels = labels

        return data, labels

    def train_model(self):
        """Train the ANN."""
        # grab data
        data, labels = self.process_data('data.npy')

        # fit the data to the model
        self.model.fit(data, labels, epochs=5, batch_size=32)

    def predict_action(self, obs):
        """Returns an action number given an observation."""
        row = []
        fs = obs.observation.feature_screen

        for screen in [fs.unit_hit_points, fs.unit_type]:
            row += list(screen.flatten())

        data = np.array(row).reshape(-1, len(row))
        return int(np.argmax(self.model.predict(data)))

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
