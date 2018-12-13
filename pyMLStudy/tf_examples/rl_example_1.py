import numpy as np
import pandas as pd
import time

# https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(num_states, actions):
    table = pd.DataFrame(np.zeros((num_states, len(actions))), columns=actions)
    return table

def choose_actions(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or (state_actions == 0).all():
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES - 2:   # terminate
            next_state = 'terminal' # right of current state is terminal
            reward = 1
        else:
            next_state = state + 1
            reward = 0

    else:
        reward = 0
        if state == 0:
            next_state = 0
        else:
            next_state = state - 1
    return next_state, reward

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0 # 初始位置
        is_terminated = False
        update_env(s, episode, step_counter)
        while not is_terminated:
            action = choose_actions(s, q_table)
            next_state, reward = get_env_feedback(s, action)
            q_predict = q_table.loc[s, action]
            if next_state != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[s, action] += ALPHA * (q_target - q_predict)
            s = next_state
            update_env(s, episode, step_counter + 1)

            step_counter += 1

    return q_table

def main():
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    return


if __name__ == '__main__':
    main()
