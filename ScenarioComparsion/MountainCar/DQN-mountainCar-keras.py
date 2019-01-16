import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from summary import *
import os
from os.path import expanduser
home = expanduser("~")

SAVE_PATH = home + "/UoA-RL.github.io/ScenarioComparsion/MountainCar/" 
NAME = "DQN Mountain car performance"
from collections import deque
import time


class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005  
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env     = gym.make("MountainCar-v0")
    gamma   = 0.9
    epsilon = .95

    trials  = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    step_summary = []
    reward_summary = []
    exTime_summary = []
    summary_types = ['sumiz_step' , 'sumiz_time', 'sumiz_reward', 'sumiz_reward_per_step']
    step_goal = 199
    reward_goal = 1
    summary_index = 5
    start_focus = 0
    end_focus = 1000
    success = 0
    for trial in range(trials):
        cur_state = env.reset().reshape(1,2)
        startTime = time.time()
        episode_count =  trial
        total_reward = 0
        step_counter = 0
        reward_counter=0
        for step in range(trial_len):
            env.render() 
            step_counter += 1 

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            total_reward += reward
             
            
            if done:
                break

        step_summary.append(step_counter)
        reward_summary.append(total_reward)
        exTime_summary.append(time.time() - startTime)
        summary(summary_types, episode_count, step_summary, exTime_summary, reward_summary, step_goal, reward_goal, summary_index, start_focus, end_focus, NAME, SAVE_PATH)
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            # if step % 10 == 0:
            #     dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            # dqn_agent.save_model("success.model")
            success += 1
            if success == 10:
                break

if __name__ == "__main__":
    main()