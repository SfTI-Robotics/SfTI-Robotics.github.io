
import numpy as numpy


import matplotlib.pyplot as plt
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image


from maze_env import Maze
from DQN_modified import DeepQNetwork
from summary import *

NAME = "Maze_DQN"
SAVE_PATH = "/Desktop/Py/Scenario_Comparasion/Maze/Model/" + "DQN_images"
summary_types = ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_average_reward']
STEP_GOAL = 6
REWARD_GOAL = 1
START_FOCUS_INDEX = 100
END_FOCUS_INDEX = 200

def run_maze():

    step = 0
    for episode in tqdm(range(200)): # for episode
        # initial observation
        observation = env.reset()
        # env.render()
        steps = 0
        reward_sum = 0
        start_time = time.time()

        while True: # for step
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)
            
            # start learning only after a certain amount of steps, in periods of 5 steps
            if (step > 20) and (step % 5 == 0):                
                RL.learn()

            # swap observation
            observation = observation_

            reward_sum += reward
            steps += 1
            step += 1
            # break while loop when end of this episode
            if done:
                break
        record.summarize(episode, steps, time.time() - start_time, reward_sum)

    # end of game
    print('game over')
    env.destroy()
    print('env destroyed')
    print('quitting...')
    env.quit()

if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=2000,
                      # output_graph=False
                      )
    record = summary(summary_types,
                    STEP_GOAL, 
                    REWARD_GOAL, 
                    START_FOCUS_INDEX, 
                    END_FOCUS_INDEX, 
                    NAME, 
                    SAVE_PATH
                    )
    env.after(100, run_maze)
    env.mainloop()
    # RL.plot_cost()
