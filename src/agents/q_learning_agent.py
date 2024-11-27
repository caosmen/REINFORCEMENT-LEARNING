import os
import pickle
import numpy as np
import gymnasium as gym

from collections import deque


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        self.n_bins = 12
        self.observation_space = [
            np.linspace(-2.4, 2.4, self.n_bins),
            np.linspace(-4, 4, self.n_bins),
            np.linspace(-0.21, 0.21, self.n_bins),
            np.linspace(-4, 4, self.n_bins)
        ]

        self.q_table = {}
        self.best_reward = -np.inf
        self.rewards_history = deque(maxlen=100)

    def discretize_state(self, state):
        discretized = []
        for i, value in enumerate(state):
            bin_index = np.digitize(value, self.observation_space[i]) - 1
            bin_index = max(0, min(bin_index, self.n_bins - 1))
            discretized.append(bin_index)
        return tuple(discretized)

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = self.discretize_state(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)

        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.env.action_space.n)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.env.action_space.n)

        next_best_action = np.argmax(self.q_table[next_state])
        current_q = self.q_table[current_state][action]
        next_max_q = self.q_table[next_state][next_best_action]

        position = state[0]
        angle = state[2]
        adjusted_reward = reward - 0.5 * (abs(position) + abs(angle))

        new_q = current_q + self.learning_rate * (
            adjusted_reward + self.discount_factor * next_max_q * (1 - done) - current_q
        )

        self.q_table[current_state][action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
