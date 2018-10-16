# Random search, hill climbing, policy gradient for CartPole

Simple reinforcement learning algorithms implemented for CartPole on OpenAI gym.

This code is part of the Move37 exercises by the school of AI [Markov Decision Process Chapter](https://www.theschool.ai/courses/move-37-course/lessons/the-bellman-equation/),

The code was inpired by [kvfrans about learning CartPole](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)
##Algorithms implemented

**Random Search**: Keep trying random weights between [-1,1] and greedily keep the best set.

**Hill climbing**: Start from a random initialization, add a little noise evey iteration and keep the new set if it improved.

**Policy gradient** Use a softmax policy and compute a value function using discounted Monte-Carlo. Update the policy to favor action-state pairs that return a higher total reward than the average total reward of that state.