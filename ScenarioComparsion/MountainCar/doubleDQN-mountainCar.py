import sys
import gym

import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

from summary import *
import os
from os.path import expanduser
home = expanduser("~")

SAVE_PATH = home + "/UoA-RL.github.io/ScenarioComparsion/MountainCar/" 
NAME = "double DQN Mountain car performance"
import time

EPISODES = 1300


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()

        # he_uniform: uniform probability distribution, weights randomly assigned, 
        #  limit is square root of (6/4)  , draws a random distribution of the limit
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform')) 
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        #  summary prints out representation of model, does not compile anything
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # behaviour choose action, target model determins Q-value based action, 
        # target model updates behavioural Q-value
        #  self.model.predict(update_input)                  

        # array of current Q-value, input is current state 
        target = self.model.predict(update_input)

        # an array of next Q-values, input next state
        target_next = self.model.predict(update_target)

        # an array of Q-values 
        #fixed q target network
        #target val is when the target model is updated to become the evaluator model
        # input is next state
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                #decouplin
                a = np.argmax(target_next[i])
                # Bellman equation
                #q value only stored ine target network not e
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        # calculates loss and does optimisation 
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('MountainCar-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []

    step_summary = []
    reward_summary = []
    exTime_summary = []
    summary_types = ['sumiz_step' , 'sumiz_time', 'sumiz_reward']
    step_goal = 200
    reward_goal = -100
    summary_index = 5
    start_focus = 0
    end_focus = 1000
    success = 0

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0
        step_counter = 0
        startTime = time.time()
        episode_count = e


        while not done:

            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            
            total_reward += reward 

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            
            state = next_state

            step_counter += 1 


            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                break

        step_summary.append(step_counter)
        reward_summary.append(total_reward)
        exTime_summary.append(time.time() - startTime)
        summary(summary_types, episode_count, step_summary, exTime_summary, reward_summary, step_goal, reward_goal, summary_index, start_focus, end_focus, NAME, SAVE_PATH)
        
        if step_counter >= 199:
            print("Failed to complete in trial {}".format(e))
        else: 
            print("Completed in {} trials".format(e), ",     {} steps taken ".format(step_counter), ",     {} rewards gained ".format(total_reward))
            success += 1
            if success == 10:
                break
