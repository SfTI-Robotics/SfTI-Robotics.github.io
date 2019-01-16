import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import os
from os.path import expanduser
from PIL import Image
home = expanduser("~")


#the focus of this class is to take in particular types of data and then transform that data into the creation of the focused and unfocused graphs associated with that data.
def summary(summary_types, episode_count, step_summary, exTime_summary, reward_summary, step_goal, reward_goal, summary_index, start_focus, end_focus, NAME, SAVE_PATH):

    #initialize the number of main axis
    num_main_axes = 0

    is_summary_r_s = False
    #looping through all the summary types in order to generate axis
    for element in summary_types:
        if element == 'sumiz_step':
            num_main_axes += 1
        if element == 'sumiz_time':
            num_main_axes += 1
        if element == 'sumiz_reward':
            num_main_axes += 1
        if element == 'sumiz_reward_per_step':
            num_main_axes += 1
            is_summary_r_s = True

    if not num_main_axes:
        return

    found_zero = False

    if len(step_summary) > 0 and len(reward_summary) > 0 and is_summary_r_s:
        for arg in step_summary:
            if arg == 0:
                found_zero = True
                num_main_axes -= 1
                print("Step array contains zero(s). reward-per-step graph will be omitted.")
                break
        if found_zero:
            reward_per_step_summary = []
        else:
            reward_per_step_summary = np.true_divide(reward_summary, step_summary)
    else:
        reward_per_step_summary = []

    if episode_count == summary_index: 
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

            if element == 'sumiz_reward_per_step' and found_zero == False:
                if step_goal != 0:
                    reward_per_step_goal = np.repeat(reward_goal/float(step_goal), len(reward_per_step_summary))
                else:
                    reward_per_step_goal = np.repeat(0, len(reward_per_step_summary))

                ax4 = fig1.add_subplot(num_main_axes, 1, i)
                ax4.plot(range(len(reward_per_step_summary)),reward_per_step_summary)
                ax4.plot(range(len(reward_per_step_summary)), reward_per_step_goal, 'r:')
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


    if episode_count == summary_index and episode_count > start_focus:
        fig2 = plt.figure(figsize=(30, 15))
        i = 1
        for element in summary_types:
            if element == 'sumiz_step':
                ax1 = fig2.add_subplot(num_focus_axes, 1, i)
                ax1.plot(range(start_focus, end_focus), step_summary[start_focus:end_focus])
                ax1.plot(range(start_focus, end_focus), np.repeat(step_goal, end_focus - start_focus), 'r:')
                ax1.set_title('Number of steps taken in each episode')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Steps taken')
                i += 1
            
            if element == 'sumiz_time':
                ax2 = fig2.add_subplot(num_focus_axes, 1, i)
                ax2.plot(range(start_focus, end_focus), exTime_summary[start_focus:end_focus])
                ax2.set_title('Execution time in each episode')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Execution time (s)')
                i += 1

        plt.tight_layout()
        fig2.savefig(SAVE_PATH + NAME +"_focused_summary.png")

        




# sumiz_step, sumiz_time, sumiz_reward, sumiz_reward_per_step