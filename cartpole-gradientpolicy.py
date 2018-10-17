import gym 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import random
import math

env = gym.make('CartPole-v0')

# Policy approach 
# a policy that can change little by little, moving from one absolute limt (move left it tot < 0
# otherwise move right) changing the agent from deterministic to stochastic policy

# Instead of only one linear combination, there are two: one for each for each possible action
# these two values are passed through a softmax function which gives the probabilities of taking
# the respective actions, given a set of obeservation, generalizing to multiplr actions, unlike the
# threshold used before  
# 


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    with tf.get_variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)

        # A way to change the policy, increasing the probability of taking certain action given a certain state
        # This is a crude implementation of an optimizer tha allows us to incremently update our policy
        # The vector actions is a one-hot encoded vector, with a one action we want to increase the probability of
        # 

        good_probabilities = tf.reduce_sum(tf.mul(probabilities,actions), reduction_indices=[1])
        # maximize the log probability
        log_probabilities = tf.log(good_probabilities)
        # insert the elementwise  maultiplication by advantages
        eligibility = log_probabilities * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return probabilities, state, actions, advantages, optimizer

    # we want to know how good it is to take action for a state. we do have a measure of success that we can make decisions
    # based on: the return or total reward from the state onwards.

def value_gradient():
    # sess.run(Calculated) to calculate value of state
    with tf.get_variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1)+b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2

        # sess.run(optimizer) to update the value of a state
        newvals = tf.placeholder("float",[None,1])
        diffs = calculated -newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

        return calculated, state, newvals, optimizer, loss


def run_episode(env, policy_grad, value_grad, sess):

    # tensorflow ops to compute probabbilities for each action, given a state
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages =[]
    transitions = []
    update_vals = []


    for _ in range(200):
        # Calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state:obs_vector})
        action = 0 if np.random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    

    
    # we then calculate the return of each transportation, and update the neural net to reflec6t this, We don't care about the specific
    # action took from each state. Only what he average return from the state over alla ctions is.
    # 


    for index, trans in enumerate(transitions):
        obs, actions, reward = trans

        # calculted discounted monte-carlo return
        future_reward = 0
        future_transition = len(transitions) - index
        decrease = 1
        for index2  in range(future_transition):
            future_reward += transitions[(index2) + index[2]] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was the action normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)


    # update the value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})


    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(vl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward




    # The decrease factor puts more of an emphasis on short-term rewrd than long term
    
    # The question the becomes how to use the newly found values in order to update our policy to reflect it.
    # We want to favour actions that return a toatl reward greater than the average of that state. This error between the 
    # actual return and the average is called an advantage. We can plug in the advantage as a scale, and update our policy
    # accordingly.

    for index, trans in enumerate(transitions):
        obs, actions, reward = trans
        # [not shown: the value function from above]
        
        
    
    