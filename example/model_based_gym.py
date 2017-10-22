# -*- coding: utf-8 -*-

"""
Gym cart pole experiment

# TODO: Currently fails to learn. Need update.
"""

from nprl import ModelBased
import gym
import numpy as np
import math
import matplotlib.pylab as plt

#  Create Environment
env = gym.make('CartPole-v0')

# observation_encoder
n_tick = np.array([1, 1, 6, 3], dtype=np.float32)
tick = env.observation_space.high - env.observation_space.low
tick[1] = 1.0
tick[3] = 2 * math.radians(50)
tick = tick / n_tick
lower_bound = env.observation_space.low
lower_bound[1] = -0.5
lower_bound[3] = -math.radians(50)


def encoder(obs):
    state = []
    for i in range(env.observation_space.shape[0]):
        j = 0
        tmp = lower_bound[i]
        if obs[i] > tmp:
            while tmp < obs[i]:
                tmp += tick[i]
                j += 1
        state.append(j)
    return str(state)


# Create Agent
agent = ModelBased(discount_factor=0.95,
                   exploration_rate=0.1,
                   exploration_reward=0.0,
                   vi_error_limit=0.01,
                   vi_iteration=100,
                   vi_interval=1)
agent.init()

obs = env.reset()
action = agent.step(new_state=encoder(obs),
                    reward=1.0,
                    terminal=False,
                    action_list=range(env.action_space.n),
                    test=False)

result = []
result_average = []
t = []
N = 1000

terminal = False
episode_time = 0
episode = 0
while True:

    # env.render()

    if terminal:
        agent.reset()

        obs = env.reset()
        reward = 1.0
        terminal = False

        # agent.exploration_rate = max(0.05, min(1.0, 1.0 - np.log10((episode+1.0)/25)))

        # Plot result
        result.append(episode_time)

        if episode % 10 == 0:
            t.append(episode)
            result_average.append(np.mean(result))
            result = []

            plt.clf()
            plt.plot(t, result_average, 'b')
            plt.hold(True)
            plt.plot([0, N], [200, 200], '-.r')
            plt.hold(False)
            plt.xlim([0, N])
            plt.ylim([0, 250])
            plt.pause(0.001)

        print("{}-th EPISODE : {} steps".format(episode, episode_time))
        episode_time = 0
        episode += 1
    else:
        obs, reward, terminal, info = env.step(action)
        print reward, terminal
        episode_time += 1

    action = agent.step(new_state=encoder(obs),
                        reward=reward,
                        terminal=terminal,
                        action_list=range(env.action_space.n),
                        test=False)

    print "action : ", action
    print "state : ", encoder(obs)
    if episode == N:
        break

print ("Close plotting window to finish.")
print "Environment Model", agent.get_model()
print "---"
print agent.get_value_function()
print "Terminal States", agent._terminal_state_set
agent.show_model()

plt.show()
print ("Finish. ")
