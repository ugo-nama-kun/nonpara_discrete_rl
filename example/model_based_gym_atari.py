# -*- coding: utf-8 -*-

"""
Gym ATARI experiment

[Currently agent fails to Learn]
"""

from nprl import ModelBased
import gym
import numpy as np
import math
import matplotlib.pylab as plt

#  Create Environment
env = gym.make('Pong-v0')

# Create Agent
agent = ModelBased(discount_factor=0.95,
                   exploration_rate=0.1,
                   exploration_reward=0.0,
                   vi_error_limit=0.01,
                   vi_iteration=100,
                   vi_interval=10,
                   maximum_state_id=10**5)
agent.init()

obs = env.reset()
action = agent.step(new_state=str(obs.tolist()),
                    reward=0.0,
                    terminal=False,
                    action_list=range(env.action_space.n))

result = []
result_average = []
t = []
N = 10**3

terminal = False
episode_time = 0
episode = 0
while True:

    #env.render()

    if terminal:
        agent.reset()

        obs = env.reset()
        reward = 0.0
        terminal = False

        agent.exploration_rate = max(0.05, min(1.0, 1.0 - np.log10((episode+1.0)/25)))

        # Plot result
        result.append(episode_time)
        t.append(episode)

        plt.clf()
        plt.plot(t, result, 'b')
        plt.hold(True)
        plt.plot([0, N], [200, 200], '-.r')
        plt.hold(False)
        plt.xlim([0, N])
        plt.ylim([-22, 22])
        plt.pause(0.001)

        print("{}-th EPISODE : {} steps".format(episode, episode_time))
        episode_time = 0
        episode += 1
    else:
        obs, reward, terminal, info = env.step(action)
        print reward, terminal
        episode_time += 1

    if episode_time >= 200 and terminal: # Pass the positive time-dependent terminal
        pass
    else:
        action = agent.step(new_state=str(obs.tolist()),
                            reward=reward,
                            terminal=terminal,
                            action_list=range(env.action_space.n))

    #print str(obs)
    print "action : ", action
    if episode == N:
        break

print ("Close plotting window to finish.")
#agent.show_model()
print len(agent.state_encoding_dict.keys())

plt.show()
print ("Finish. ")
