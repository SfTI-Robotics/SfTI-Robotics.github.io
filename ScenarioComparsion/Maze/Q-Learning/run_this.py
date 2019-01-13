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
from RL_brain import QLearningTable
import matplotlib.pyplot as plt
import os
from os.path import expanduser
from PIL import Image
import time
home = expanduser("~")

SAVE_PATH = home + "/Desktop/Py/Scenario_Comparasion/Maze/Model/" + "QL_images/"

def update():
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    filelist = [ f for f in os.listdir(SAVE_PATH) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(SAVE_PATH, f))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ep_steps=[0]
    ep_times=[0]
    ep_rewards=[0]

    for episode in range(200):
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

        ep_steps=np.append(ep_steps,[steps])
        ep_times=np.append(ep_times,[time.time() - start_time])
        ep_rewards=np.append(ep_rewards,[reward_sum])

        if episode % 3 == 0 and episode > 100:
            fig1,(ax1, ax2)=plt.subplots(2,1,sharex=False)
            ax1.plot(range(len(ep_steps) - 100),ep_steps[100:])
            ax1.set_title('Number of steps taken in each episode')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Steps taken')
            
            ax2.plot(range(len(ep_times) - 100),ep_times[100:])
            ax2.set_title('Execution time in each episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Execution time (s)')

            plt.tight_layout()
            fig1.savefig(SAVE_PATH + "focus_" + str(episode)+".png")

        if episode % 3 == 0: 
            fig1,(ax1, ax2, ax3)=plt.subplots(3,1,sharex=False)
           
            ax1.plot(range(len(ep_steps)),ep_steps)
            ax1.set_title('Number of steps taken in each episode')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Steps taken')
            
            ax2.plot(range(len(ep_times)),ep_times)
            ax2.set_title('Execution time in each episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Execution time (s)')

            ax3.plot(range(len(ep_rewards)),ep_rewards)
            ax3.set_title('Reward in each episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')

            plt.tight_layout()
            fig1.savefig(SAVE_PATH + str(episode)+".png")

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    s = time.time()
    env = Maze()
    print("1. ", time.time() - s)

    s = time.time()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    print("2. ", time.time() - s)

    s = time.time()
    env.after(100, update)
    print("3. ", time.time() - s)

    s = time.time()
    env.mainloop()
    print("4. ", time.time() - s)