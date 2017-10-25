# -*- coding: utf-8 -*-
"""
Python implementation of Q-learning for discrete-finite MDPs
Naoto Yoshida
"""

import random
import numpy as np


class QLearning(object):
    def __init__(self,
                 lr=0.05,
                 discount_factor=0.95,
                 exploration_rate=0.1,
                 exploration_rate_test=0.05,
                 initial_q_value=0.0,
                 initial_fluctuation=True,
                 maximum_state_id=10000):
        """
        Q-Learning Agent
        :param float lr: learning rate [0, 1)
        :param float discount_factor: discount factor [0, 1)
        :param float exploration_rate: exploration rate for the epsilon-greedy action selection [0, 1]
        :param float exploration_rate_test: exploration rate for the epsilon-greedy action selection [0, 1] in TEST
        :param float initial_q_value: (mean) initial action value for state-action pair
        :param bool initial_fluctuation: add small fluctuation (-10**-5 ~ 10**-5) to initialize action value or not
        :param int maximum_state_id: maximum number of states that the agent generates
        """

        # Value Parameters
        self._q_values = {} # {state: {action: value}}
        self._state_encoding_dict = {} # {state: tag}
        self.lr = lr
        self._discount_factor = discount_factor
        self._initial_q_val = initial_q_value
        self._maximum_state_id = maximum_state_id
        self.initial_fluctuation = initial_fluctuation

        # Policy parameters
        self.exploration_rate = exploration_rate
        self.exploration_rate_test = exploration_rate_test

        # Initialization of parameters used in the algorithm
        self._state_action = None

    def init(self):
        """
        Initialize agent to the initial status
        :return:
        """
        self._q_values = {}
        self._state_action = None

    def reset(self):
        """
        Reset agent to the initial status for new episode
        :return:
        """

        self._state_action = None

    def _get_max_q(self, state):
        """
        Obtain the maximum action value and action at the given state
        :param state:
        :return: float action_value, corresponding action, action-value dictionary
        """
        action_and_q_dict = self._q_values[state]

        action_and_q_dict.values()
        max_val = np.max(action_and_q_dict.values())
        best_action = action_and_q_dict.keys()[np.argmax(action_and_q_dict.values())]
        return max_val, best_action, action_and_q_dict

    def _get_action(self, state, action_list, test=False):
        """
        Obtain next action with respect to the given state
        :param state: current state
        :param action_list: action candidates at the given state
        :param bool test: switch the agent take test behavior or learning behavior
        :return:
        """
        if test:
            eps = self.exploration_rate_test
        else:
            eps = self.exploration_rate
            
        if random.random() < eps:
            return random.choice(action_list)
        else:
            _, best_action, action_q_dict = self._get_max_q(state)
            return best_action

    def _add_new_state_action_if_unknown(self, state, action_list):
        """
        Add new state if the given state is new to the agent
        :param state:
        :param action_list:
        """
        if not self._match_state(state):
            self._q_values.update({state: {}})
            for action in action_list:
                f = 0.0
                if self.initial_fluctuation:
                    f = np.random.uniform(low=-10**-5, high=10**-5)
                self._q_values[state].update({action: self._initial_q_val + f})
        else:
            # State found but action_list is different from previous status
            for action in action_list:
                if not self._match_action(state, action):
                    f = 0.0
                    if self.initial_fluctuation:
                        f = np.random.uniform(low=-10 ** -5, high=10 ** -5)
                    self._q_values[state].update({action: self._initial_q_val + f})

    def _match_state(self, state):
        """
        Hard matching of the state.
        :param state:
        :return: bool: True if state is matched in the history
        """
        return state in self._q_values.keys()

    def _match_action(self, state, action):
        """
        Hard matching of the action given a state.
        :param state:
        :param action:
        :return: bool: True if state is matched in the history
        """
        return action in self._q_values[state].keys()

    def _set_terminal_value(self, state, action_list):
        """
        Set terminal value (zero) at terminal states
        :param state:
        :param list action_list: action set at new_state A(s)
        """
        for action in action_list:
            self._q_values[state].update({action: 0.0})

    def _update_q(self, state_action, next_state, reward, terminal):
        """
        Update value function by simple TD-learning algorithm
        :param tuple state_action:
        :param next_state:
        :param float reward: reward as a result of (state, action, new_state) triple
        :param bool terminal: Terminal flag
        """

        state = state_action[0]
        action = state_action[1]

        if terminal is False:
            max_q, _, _ = self._get_max_q(next_state)
            td_error = float(reward) + self._discount_factor*max_q - self._q_values[state][action]
        else:
            td_error = float(reward) - self._q_values[state][action]

        new_value = self._q_values[state][action] + self.lr * td_error
        self._q_values[state][action] = new_value

    def _learning_step(self, new_state, new_action, reward, terminal, action_list):
        """
        Proceed single step
        :param new_state: state observed by agent
        :param new_action: action taken with respect to new_state
        :param float reward: reward as a result of (state, action, new_state) triple
        :param bool terminal: Terminal flag
        :param list action_list: action set at new_state A(s)
        :return: 
        """

        # Assert is number of environment state is larger than the expected maximum_id
        if self._maximum_state_id is not None:
            assert len(self._q_values.keys()) < self._maximum_state_id, "Too many state! |S| > " + str(self._maximum_state_id)

        # Set previous state_action
        if self._state_action is None:
            self._state_action = (new_state, new_action)
        else:
            self._update_q(self._state_action, new_state, reward, terminal)
            self._state_action = (new_state, new_action)

        # Treatment of terminal states
        if terminal:
            self._set_terminal_value(new_state, action_list)
            self._state_action = None

    def _encode_state(self, state):
        if state in self._state_encoding_dict.keys():
            return self._state_encoding_dict[state]
        else:
            tag = len(self._state_encoding_dict.keys())
            self._state_encoding_dict.update({state : tag})
            return tag

    def step(self, new_state, reward, terminal, action_list, test=False):
        """
        Take a sigle step of the agent
        :param new_state: state observed by agent
        :param float reward: reward as a result of (state, action, new_state) triple
        :param bool terminal: Terminal flag
        :param list action_list: action set at new_state A(s) as a list of strings
        :param bool test:
        :return:
        """
        assert len(action_list) > 0, "action_list has to have at least one action"

        # State encoding
        new_state_ = self._encode_state(new_state)

        # Expand dictionary if new_state is unknown
        if not test:
            self._add_new_state_action_if_unknown(new_state_, action_list)

        # Take an action
        new_action = self._get_action(new_state_, action_list, test)

        if not test:
            # Update Stats
            self._learning_step(new_state_, new_action, reward, terminal, action_list)
        return new_action

    def get_q_value(self):
        """
        Obtain Value Function
        :return: value dictionary
        """
        return self._q_values
