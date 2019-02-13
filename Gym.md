# OpenAI gym/retro
## Open environment
make(): 
```
env = gym.make('[insert name of env]') # for gym
env = retro.make('[insert name of env]') # for retro
```

```
env.observation_space.shape[0] # continous
env.observation_space.n # discrete
action_size = env.action_space.n

```

state_size = env.observation_space.n

## Retrieve observation
reset(): returns an initial observation/state
```
observation = env.reset() # often also used to return state
```
## Perform action
step(): 
```
observation_next, reward, done, info = env.step(action) 
```

## Destroy environment 
close(): prevents data leaks → ensures code runs every time without reopening terminal
```
env.close() 
```
## Wrappers
Monitor(): 

## Spaces

