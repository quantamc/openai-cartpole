import gym
import numpy as np
import matplotlib.pyplot as plt 


env = gym.make('CartPole-v0')

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        # each weight is multiplied by its respective observatio, and the products are summed up
        # if total is less than 0, we move left. Otherwise we move right
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


# Random serach, try random weights and pick one that performs the best
def train(submit):
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        #initialize a weight vector
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                break

    if submit:
        for _ in range(100):
            run_episode(env, bestparams)

        env.monitor.close()

    return counter

# train agent to submit to openai gym
# train(submit=True)

# create graphs
results = []
for _ in range(1000):
    results.append(train(submit=False))
print(results)
plt.hist(results,50,normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

print(np.sum(results) / 100.0)