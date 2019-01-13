
import numpy as numpy
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from os.path import expanduser
from PIL import Image
home = expanduser("~")

from maze_env import Maze
from DQN_modified import DeepQNetwork

SAVE_PATH = home + "/Desktop/Py/Scenario_Comparasion/Maze/Model/" + "DQN_images/"

def run_maze():
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    filelist = [ f for f in os.listdir(SAVE_PATH) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(SAVE_PATH, f))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ep_steps=[0]
    ep_times = [0]
    ep_rewards=[0]

    step = 0
    for episode in range(200): # for episode
        # initial observation
        observation = env.reset()
        # env.render()
        reward_sum = 0
        step_counter = 0
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
            step += 1
            step_counter += 1
            
            # break while loop when end of this episode
            if done:
                break
            
            

        ep_steps=np.append(ep_steps,[step_counter])
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
            fig1,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=False)
            ax1.plot(range(len(ep_steps)),ep_steps)
            ax1.set_title('Number of steps taken in each Episode')
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
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
