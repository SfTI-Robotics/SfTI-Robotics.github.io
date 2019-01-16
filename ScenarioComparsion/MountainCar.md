# Mountain Car Scenario Comparsion

## Mountain Car Enviornment
The mountain car environment involves a left mountain and right mountain that has a flag at position 0.5(and is the goal
). The episode completes when the car reaches the flag but must complete this task in 200 time steps or it has failed. The agent starts in the midddle of the two mountains between a random position of  -0.6 to -0.4 with no velocity. If the car goes beyond the left mountainit also fails. The total reward it can accumalate is 1 which is given at the flag position only but recieves -1 for each  time step, until the goal position of 0.5 is reached. As with MountainCarContinuous v0, there is no penalty for climbing the left hill. The states that can be observed are its position from  -1.2 to 0.6 and velocity from -0.07 to 0.07. The actions the car can take is push left, right or no force.

![Maze](/MountainCar.png)


## Steps per Episode Comparsion
The difference between DQN and Q - learning algorithms is notified through the osciallations produced in the normalisation of the optimal policy in the later episodes. This due to the exploration and exploitation comparative. As the Q - learning algorithm explores and exploits at the same time on the same network while the DQN has a two simultaneous networks one of which allows the DQN to explore while the target newtork allows it to exploit and choose an optimal policy. 

### QL focus Graphs
![Graph](Maze/focus_QL.png)

### DQN focus Graphs 
![Graph](Maze/focus_DQN.png)



## Conclusion
The Q - learning provides a slight oscillation which is a negligible difference in the steps which are produced the runtimes are also quite similar seen in the graphs above there we can conclude it wouldn't provide much of a difference in the choosing of one algorithm over the other but for a slightly more normalised optimal policy generation the DQN could be chosen.
