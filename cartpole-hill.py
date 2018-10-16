import gym 
import numpy as np 
import matplotlib.pyplot as plt 

env = gym.make('CartPole-v0')

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0
    for _ in range(200):
        # each weight is multiplied by its respective observatio, and the products are summed up
        # if total is less than 0, we move left. Otherwise we move right
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        counter += 1
        if done:
            break
    return totalreward


####### Hill climbing
# start eith random choosen weights, every episode, add
# some noise to the weights and keep the new weights if the agent improves

# gradiually improve the weights, rather than jumping around and hopefully
# find some combination that works

def train(submit):
    if submit:
        env.monitor.start('cartpole-hill/', force=True)

    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2  - 1
    bestreward = 0
    counter = 0

    for _ in range(2000):
        counter += 1
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling

        print(newparams)

        reward = run_episode(env, newparams)
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break

    if submit:
        for _ in range(100):
            run_episode(env,parameters)
        env.monitor.close()
    return counter

# create graphs
results = []
for _ in range(1000):
    results.append(train(submit=False))
plt.hist(results,50,normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Hill Climbing')
plt.show()

print(np.sum(results) / 100.0)