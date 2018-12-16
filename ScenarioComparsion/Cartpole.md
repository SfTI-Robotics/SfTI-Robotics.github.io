# Cartpole
The code resources used in this comparison can be found [in this repo](https://github.com/UoA-RL/Gym/tree/master/Q-Learning) and in this [page](https://github.com/UoA-RL/UoA-RL.github.io/blob/master/Code_Comparison/QNetwork_comparison.md). 


# Q networks
- initialise Q-values in a table matrix using Numpy
- from observation, find state using angle to buckets (what is the difference)


# DQN networks
- use the reshaped vector from `env.reset()` as state 
- uses neural networks to calculate Q-values 