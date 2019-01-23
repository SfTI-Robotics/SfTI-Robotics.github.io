import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import os
from os.path import expanduser
from PIL import Image
home = expanduser("~")


INDEX = 199
FOCUS_INDEX = 100
NAME = "Maze_QL"
SAVE_PATH = home + "/Desktop/Py/Scenario_Comparasion/Maze/Model/" + "DQN_images/"


def summary(summary_types, episode_count, step_summary = [], exTime_summary = [], reward_summary = [], step_goal = 0, reward_goal = 0):
    num_summaries = len(summary_types)
    num_main_axes = 0

    for element in summary_types:
        if element == 'sumiz_step':
            num_main_axes += 1
        if element == 'sumiz_time':
            num_main_axes += 1
        if element == 'sumiz_reward':
            num_main_axes += 1
        if element == 'sumiz_reward_per_step':
            num_main_axes += 1

    if not num_main_axes:
        return

    if len(step_summary) > 0 and len(reward_summary) > 0 and not np.argmin(step_summary):
        reward_per_step_summary = np.true_divide(reward_summary, step_summary)
    else:
        reward_per_step_summary = []

    if episode_count == INDEX: 
        fig1 = plt.figure(figsize=(30, 15))
        i = 1
        for element in summary_types:
            if element == 'sumiz_step':
                ax1 = fig1.add_subplot(num_main_axes, 1, i)
                ax1.plot(range(len(step_summary)),step_summary)
                ax1.plot(range(len(step_summary)),np.repeat(step_goal, len(step_summary)), 'r:')
                ax1.set_title('Number of steps taken in each episode')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Steps taken')
                i += 1
            
            if element == 'sumiz_time':
                ax2 = fig1.add_subplot(num_main_axes, 1, i)
                ax2.plot(range(len(exTime_summary)),exTime_summary)
                ax2.set_title('Execution time in each episode')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Execution time (s)')
                i += 1

            if element == 'sumiz_reward':
                ax3 = fig1.add_subplot(num_main_axes, 1, i)
                ax3.plot(range(len(reward_summary)),reward_summary)
                ax3.plot(range(len(reward_summary)), np.repeat(reward_goal, len(reward_summary)), 'r:')
                ax3.set_title('Reward in each episode')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Reward')
                i += 1

            if element == 'sumiz_reward_per_step':
                ax4 = fig1.add_subplot(num_main_axes, 1, i)
                ax4.plot(range(len(reward_per_step_summary)),reward_per_step_summary)
                ax4.plot(range(len(reward_per_step_summary)), np.repeat(reward_goal/float(step_goal), len(reward_per_step_summary)), 'r:')
                ax4.set_title('Reward in each episode per step')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Reward per step')
                i += 1

        plt.tight_layout()
        fig1.savefig(SAVE_PATH + NAME + "_summary.png")

    num_focus_axes = 0
    for element in summary_types:
        if element == 'sumiz_step':
            num_focus_axes += 1
        if element == 'sumiz_time':
            num_focus_axes += 1

    if not num_focus_axes:
        return


    if episode_count == INDEX and episode_count > FOCUS_INDEX:
        fig2 = plt.figure(figsize=(30, 15))
        i = 1
        for element in summary_types:
            if element == 'sumiz_step':
                ax1 = fig2.add_subplot(num_focus_axes, 1, i)
                ax1.plot(range(FOCUS_INDEX, len(step_summary)),step_summary[FOCUS_INDEX:])
                ax1.plot(range(FOCUS_INDEX, len(step_summary)),np.repeat(step_goal, len(step_summary) - FOCUS_INDEX), 'r:')
                ax1.set_title('Number of steps taken in each episode')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Steps taken')
                i += 1
            
            if element == 'sumiz_time':
                ax2 = fig2.add_subplot(num_focus_axes, 1, i)
                ax2.plot(range(FOCUS_INDEX, len(exTime_summary)),exTime_summary[FOCUS_INDEX:])
                ax2.set_title('Execution time in each episode')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Execution time (s)')
                i += 1

        plt.tight_layout()
        fig2.savefig(SAVE_PATH + NAME +"_focused_summary.png")

        




# sumiz_step, sumiz_time, sumiz_reward, sumiz_reward_per_step