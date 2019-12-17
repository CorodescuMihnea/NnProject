"""
    Implementation of a reinforcement learning agent using neural nets
    Based on the example at:
        https://blog.dominodatalab.com/deep-reinforcement-learning/
"""
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from collections import deque

import random as rand
import numpy as np

class DeepQAgent:
    def __init__(self, state_space_size, action_space_size, model=1, is_discrete_action_space=True):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.is_discrete_action_space = is_discrete_action_space

        # Exploration rate (i.e. nr of random actions or how often the agent uses its acquired knowledge)
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .9

        # Discount factor (lowers future reward)
        self.gamma = .95

        # Stochastic gradient descent hyperparam
        self.learning_rate = 0.001

        # Memory of actions taken
        # Size is nof_episodes * batch_size
        self.memory = deque(maxlen=32000)

        if model == 1:
            self.model = self._build_model1()
        elif model == 2:
            self.model = self._build_model2()
        else:
            raise NotImplementedError

    def _build_model1(self):
        model = models.Sequential()
        model.add(layers.Dense(32, input_dim=self.state_space_size, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        # Directly oututing z seems to be recommended for DQNs
        model.add(layers.Dense(self.action_space_size, activation="linear"))
        # Adam seems to be the go to optimizer choice in most examples
        optimizer = optimizers.Adam(lr=self.learning_rate)
        # Mse is good for linear output layers
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def _build_model2(self):
        model = models.Sequential()
        model.add(layers.Dense(512, input_dim=self.state_space_size, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        # Directly oututing z seems to be recommended for DQNs
        model.add(layers.Dense(self.action_space_size, activation="linear"))
        # Adam seems to be the go to optimizer choice in most examples
        optimizer = optimizers.Adam(lr=self.learning_rate)
        # Mse is good for linear output layers
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        # Currently it is possible for a time_step(frame) to be used twice in training
        minibatch = rand.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # if done
            if not done:
                # print(state)
                # print(next_state)
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        # If epsilon(exploration factor) hasn't decayed enough most of the actions will be random
        if np.random.rand() <= self.epsilon:
            if self.is_discrete_action_space:
                return rand.randint(0, self.action_space_size - 1)
            else:
                raise NotImplementedError
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)