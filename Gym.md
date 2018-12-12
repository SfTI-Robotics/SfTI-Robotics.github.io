#OpenAI gym/retro
##Open environment
make(): 
```
env = gym.make('[insert name of env]') # for gym
env = retro.make('[insert name of env]') # for retro
```

##Retrieve observation
reset(): returns an initial observation/state
```observation = env.reset() # often also used to return state
```
##Perform action
step(): 
```observation_next, reward, done, info = env.step(action) 
```

##Destroy environment 
close(): prevents data leaks → ensures code runs every time without reopening terminal
```env.close() 
```
