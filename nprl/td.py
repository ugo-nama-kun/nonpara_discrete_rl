# -*- coding: utf-8 -*-
"""
Python implementation of TD Learning (Experimental) for discrete-finite MDPa
Naoto Yoshida
"""


class TDLearning(object):
    def __init__(self,
                 lr=0.05,
                 discount_rate=0.95,
                 initial_v_value=0.0,
                 maximum_state_id=10000):

        self._values = {} # {state: value}
        self._state_encoding_dict = {} # {state: tag}
        self._lr = lr
        self._discount = discount_rate
        self._initial_v_val = initial_v_value
        self._maximum_state_id = maximum_state_id

        self._prev_state = None

    def step(self, state, reward, terminal):
        """
        Proceed single step
        :param state: any state
        :param float reward: reward
        :param bool terminal: Terminal flag
        """

        # State encoding
        state_ = self._encode_state(state)

        self._add_new_state_if_unknown(state_)

        # Assert is number of environment state is larger than the expected maximum_id
        if self._maximum_state_id is not None:
            assert len(self._values.keys()) < self._maximum_state_id, "Too many state! |S| > " + str(self._maximum_state_id)

        # Set previous state
        if self._prev_state is None:
            self._prev_state = state_
        else:
            self._update_td(self._prev_state, state_, reward, terminal)
            self._prev_state = state_

        # Treatment of terminal states
        if terminal:
            self._set_terminal_value(state_)
            self._prev_state = None

    def _add_new_state_if_unknown(self, state):
        """
        Add new state if the given state is new to the agent
        :param state:
        """
        if not self._match(state):
            self._values[state] = self._initial_v_val

    def _set_terminal_value(self, state):
        """
        Set terminal value (zero) at terminal states
        :param state:
        """
        self._values[state] = 0.0

    def _encode_state(self, state):
        if state in self._state_encoding_dict.keys():
            return self._state_encoding_dict[state]
        else:
            tag = len(self._state_encoding_dict.keys())
            self._state_encoding_dict[state] = tag
            return tag

    def _update_td(self, state, state_dash, reward, terminal):
        """
        Update value function by simple TD-learning algorithm
        :param state:
        :param state_dash:
        :param reward:
        :param terminal:
        """

        if terminal is False:
            td_error = float(reward) + self._discount*self._values[state_dash] - self._values[state]
        else:
            td_error = float(reward) - self._values[state]

        new_value = self._values[state] + self._lr * td_error
        self._values[state] = new_value

    def _match(self, state):
        """
        Hard matching of the state
        :param state: some object
        :return: bool: True if state is matched in the history
        """
        return state in self._values.keys()

    def get_v_value(self):
        """
        Obtain Value Function
        :return: value dictionary
        """
        return self._values