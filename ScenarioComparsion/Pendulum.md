# Pendulum Comparison

Pendulum is a classic control environment which aims for the pole to remain upright vertically at 0 degrees. There are no episodic terminations, instead, only the steps and memory are taken into account. The reward gained for each action when there is no movement(i.e: pendulum is already upright) is 0, and for each time step, a reward of -1 is assigned. The lowest reward for each action is -16.2736044 since the pendulum can only move from -pi to pi. In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity, and the least effort.Also note that there is no reward threshold to say that it has successfully achieved the goal.

## Results

The graphs for the Q-evaluation value, cost and accumulated reward was graphed against the number of steps. The code was run for --- steps. 

We hypothesised that dueling DQN would result in a higher accumulative reward and also reach a lower cost faster than double DQN, signifying that it was faster at learning. 


#### Double DQN 
![Graph](


#### Dueling DQN
![Graph](


## Conclusions