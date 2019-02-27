"""
Code uses objects and functions
"""

import random
import gym
import numpy as np
import time
from summary import *
import os
import keras

#  python module for array manipulation
from collections import deque 
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from tqdm import tqdm
# from gym_recording.wrappers import TraceRecordingWrapper


# from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

EPISODE = 500
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 10000
TRAIN_START = 1000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999

summary_types = ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_epsilon']
STEP_GOAL = 500
REWARD_GOAL = 500
EPSILON_GOAL = 0.99
SUMMARY_INDEX = 100
start_focus = 0
end_focus = 0
NAME = 'CartPole-v1_DQN'
SAVE_PATH = "/"

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.observation_space = observation_space
        # initialise memory array with a 'deque' data structure
        self.memory = deque(maxlen=MEMORY_SIZE)

        # create neural network by stacking the layers in a linear order
        self.model = Sequential()
        # self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        # self.model.add(Dense(24, activation="relu"))
        # self.model.add(Dense(self.action_space, activation="linear"))
        # # if weights are not specified, default is sample weight
        # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        DQNSolver.build_network22(self)

    def build_network22(self):
        shape_image = self.observation_space
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        #self.model.add(keras.layers.Lambda(lambda x: x / 255.0,input_shape = shape_image))
        self.model.add(Dense(512, activation="relu", kernel_initializer='he_uniform' ))
        self.model.add(Dense(24, activation="relu", kernel_initializer='he_uniform' ))
        self.model.add(Dense(self.action_space, activation="linear", kernel_initializer ='he_uniform'))
        self.model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00025, rho = 0.95), loss="mse")

    # store transition
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # choose action 
    def act(self, state):
        # rand: create array of given shape and populate with random samples from a uniform distribution over (0, 1)
        if np.random.rand() < self.exploration_rate: # exploration
            #  randrange: generate numbers from a range, allows intervals between numbers
            return random.randrange(self.action_space)
        #  predict: computation performed in batches, updates q-values
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) # exploitation

    def experience_replay(self):
        
        if len(self.memory) < TRAIN_START:
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        # ======================================================
        # for state, action, reward, state_next, done in batch:

        #     target = reward
        #     if not done:
        #         # resize array by increasing dimension
        #         #state_next = np.expand_dims(state_next, axis= 0)
        #         # bootstrapping the predicted reward as Q-value
        #         target = reward + GAMMA * np.max(self.model.predict(state_next))

        #     # resize array by increasing dimension
        #     #state = np.expand_dims(state, axis=0)
        #     target_f = self.model.predict(state)

        #     target_f[0][action] = target
        #     # print('target_f =', target_f)
        #     self.model.fit(state, target_f,batch_size=BATCH_SIZE,epochs=1, verbose = 0)

         # ======================================================
        update_input = np.zeros((BATCH_SIZE, self.observation_space))
        update_target = np.zeros((BATCH_SIZE, self.observation_space))
        action, reward, done = [], [], []

        for i in range(BATCH_SIZE):
            update_input[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            update_target[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.model.predict(update_input)

        target_next = self.model.predict(update_target)

        for i in range(BATCH_SIZE):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # bellman equation
                target[i][action[i]] = reward[i] + GAMMA * np.amax(target_next[i])

        self.model.fit(update_input, target, batch_size=BATCH_SIZE, epochs=1, verbose=0)

         # ======================================================
        # state_array = np.zeros((BATCH_SIZE, self.observation_space)) 
        # next_state_array = np.zeros((BATCH_SIZE, self.observation_space))
        # print("observation =", self.observation_space)
        # out = np.zeros((BATCH_SIZE,2))
        # # batch = random.sample(self.transitions, batch_size)
        # i = 0
        # for state, action, reward, state_next, terminal in batch:
        #     state_array[i:i+1] = state
        #     next_state_array[i:i+1] = state_next
        #     target = reward

        #     if terminal == False:
        #         #state_next = np.expand_dims(state_next, axis=0)
        #         target = reward + GAMMA * np.amax(self.model.predict(state_next)[0] )
            
        #     #state = np.expand_dims(state, axis=0)
        #     out[i] = self.model.predict(state)
        #     out[i][action] = target
        #     i += 1

        # self.model.fit(state_array,out,batch_size = BATCH_SIZE, epochs=1,verbose=1)

         # ======================================================
        print("finish replay")
        # update exploration rate
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        print('e =', 1 - self.exploration_rate)



def cartpole():
    env = gym.make(ENV_NAME)
    # env = TraceRecordingWrapper(env)

    # print(env.directory)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    # for episode_counter in tqdm(range(200)):
    for ep in tqdm(range(EPISODE)):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        start_time = time.time()
        reward_sum = 0

        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            # Question: why reward = 0 at end of episode
            reward = reward if not terminal else -reward
            
            # reshape state_next data into 1 by states matrix
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next

            reward_sum += reward

            if terminal:
                # print "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


        record.summarize(ep, step, time.time() - start_time, reward_sum, 1 - float(dqn_solver.exploration_rate))

    #end of game
    print('game over')
    env.destroy()
    print('env destroyed')
    print('quitting....')
    env.quit()

if __name__ == "__main__":
    record = summary(summary_types,STEP_GOAL, REWARD_GOAL, EPSILON_GOAL, start_focus, end_focus, NAME, SAVE_PATH)
    cartpole()