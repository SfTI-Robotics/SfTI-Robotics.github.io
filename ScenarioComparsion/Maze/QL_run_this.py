"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
from maze_env import Maze
from QL_RL_brain import QLearningTable
from summary import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

NAME = "Maze_QL"
SAVE_PATH = "/Desktop/Py/Scenario_Comparasion/Maze/Model/" + "QL_images/"
STEP_GOAL = 6
REWARD_GOAL = 1
EPSILON_GOAL = 0.9
START_FOCUS_INDEX = 100
END_FOCUS_INDEX = 200
summary_types = ['sumiz_step', 'sumiz_reward', 'sumiz_time', 'sumiz_epsilon']

def update():

    record.display_parameters(intial_epsilon = 0.9, max_epsilon = 0.9, learning_rate = 0.01, reward_decay = 0.9)

    for episode in tqdm(range(200)):
        # initial observation
        observation = env.reset()
        env.render()
        steps = 0
        reward_sum = 0
        start_time = time.time()
        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            steps += 1
            reward_sum += reward
            # break while loop when end of this episode
            if done:
                break    

        record.summarize(episode, steps, time.time() - start_time, reward_sum)

    # end of game
    print('game over')
    env.destroy()
    print('environment destoryed')
    print('quitting...')
    env.quit()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    record = summary(summary_types,
                    STEP_GOAL, 
                    REWARD_GOAL, 
                    EPSILON_GOAL,
                    START_FOCUS_INDEX, 
                    END_FOCUS_INDEX, 
                    NAME, 
                    SAVE_PATH
                    )
    env.after(100, update)
    env.mainloop()
