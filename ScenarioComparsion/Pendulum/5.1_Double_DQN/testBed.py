"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from DoubleDQN import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from summary import summary
import time

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

summary_types = ['sumiz_time', 'sumiz_reward']
step_goal = 10000
reward_goal = 0
start_focus = 0
end_focus = step_goal
NAME = "DoubleDQN_Pendulum"
SAVE_PATH = "/UoA-RL.github.io/ScenarioComparsion/Pendulum/5.1_Double_DQN/"
start_time = time.time()

results = summary(summary_types, step_goal, reward_goal, start_focus, end_focus, NAME, SAVE_PATH)

sess = tf.Session()
with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1

        results.summarize(total_steps, time_count = start_time - time.time(), reward_count = reward)
    return RL.q

q_double = train(double_DQN)



plt.plot(np.array(q_double), c='b', label='double')

plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
