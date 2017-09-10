# -*- coding: utf-8 -*-

from nprl.q_learning import QLearning


# Environment : T-maze
# # -1 for state 0, +1 for state 2. 5 is the initial state.
#
# 0, 1, 2
#    3
#    4
#    5
# Action : [up, down, right, left]

class GridEnv(object):
    def __init__(self):
        self.action_list = ["up", "down", "right", "left"]
        self.state = 5

    def reset(self):
        self.state = 5
        return self.state, 0.0, False

    def step(self, action):
        terminal = False
        reward = 0.0
        if self.state == 5:
            if action == "up":
                self.state = 4
        elif self.state == 4:
            if action == "up":
                self.state = 3
            elif action == "down":
                self.state = 5
        elif self.state == 3:
            if action == "up":
                self.state = 1
            elif action == "down":
                self.state = 4
        elif self.state == 1:
            if action == "down":
                self.state = 3
            elif action == "right":
                self.state = 2
                reward = 1.0
                terminal = True
                print("Success")
            elif action == "left":
                self.state = 0
                reward = -1.0
                terminal = True
                print("Failure")

        return self.state, reward, terminal

if __name__ == '__main__':
    # Create Agent
    qlean = QLearning(initial_q_value=0.0,
                      exploration_rate=1.0,
                      lr=0.01,
                      discount_factor=0.5,
                      initial_fluctuation=True)
    qlean.reset()

    # Create Environment
    env = GridEnv()

    # Start Experiment
    print("Start Grid Experiment")
    state, reward, terminal = env.reset()

    action = qlean.step(new_state=state,
                        reward=reward,
                        terminal=terminal,
                        action_list=env.action_list,
                        test=False)
    terminal = False
    episode_time = 0
    episode = 1
    while episode < 1000:

        if terminal:
            # Reset agent for the next episode
            qlean.reset()

            # Reset environment
            state, reward, terminal = env.reset()

            print("{}-th EPISODE : {} steps".format(episode, episode_time))

            episode_time = 0
            episode += 1

            qlean.exploration_rate -= 10 ** -3
            if qlean.exploration_rate < 0.01:
                qlean.exploration_rate = 0.01
        else:
            state, reward, terminal = env.step(action)
            episode_time += 1

        # Single Update
        action = qlean.step(new_state=state,
                            reward=reward,
                            terminal=terminal,
                            action_list=env.action_list,
                            test=False)

    print "Finish"
    print qlean.get_q_value()