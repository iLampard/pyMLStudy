import numpy as np
import pandas as pd


class QLearningTable(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_actions(self, observation):
        self.check_state_exists(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, next_s):
        self.check_state_exists(next_s)
        q_predict = self.q_table.loc[s, a]
        if next_s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[next_s, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
