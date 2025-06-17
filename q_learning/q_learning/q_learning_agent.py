from pathlib import Path

q_learning_code = '''\
# q_learning_agent.py

import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, env, state_size, action_size, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon_start    # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.epsilon_decay = epsilon_decay

        self.q_table = {}               # Dictionary-based Q-table

    def _discretize_state(self, state):
        """Convert continuous state to discrete for Q-table indexing."""
        return tuple(np.round(state, 3))  # e.g., [0.12, 0.98] → (0.12, 0.98)

    def act(self, state):
        """Epsilon-greedy action selection."""
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state_key]))

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        state_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        best_next_q = np.max(self.q_table[next_key])
        target = reward + self.gamma * best_next_q * (1 - int(done))
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

        # Decay epsilon only at end of episode
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="models/q_table.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f" Q-table saved to {path}")

    def load(self, path="models/q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f" Q-table loaded from {path}")
        else:
            print(f" Q-table file not found at: {path}")
'''

# Save the code to file
