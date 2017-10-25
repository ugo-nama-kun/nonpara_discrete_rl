# -*- coding: utf-8 -*-
"""
Python implementation of TD Learning (Experimental) for discrete-finite MDPa
Naoto Yoshida
"""


class TDLearning(object):
    def __init__(self,
                 alpha=0.05,
                 gamma=0.95,
                 initial_v_value=0.0,
                 maximum_state_id=10000):

        self.__values = {} # {state: value}
        self._state_encoding_dict = {} # {state: tag}
        self.__alpha = alpha
        self.__gamma = gamma
        self.__initial_v_val = initial_v_value
        self.__maximum_state_id = maximum_state_id

        self.__prev_state = None

    def step(self, state, reward, terminal):
        """
        Proceed single step
        :param state: any state
        :param float reward: reward
        :param bool terminal: Terminal flag
        """

        # State encoding
        state_ = self._encode_state(state)

        self.__add_new_state_if_unknown(state_)

        # Assert is number of environment state is larger than the expected maximum_id
        if self.__maximum_state_id is not None:
            assert len(self.__values.keys()) < self.__maximum_state_id, "Too many state! |S| > " + str(self.__maximum_state_id)

        # Set previous state
        if self.__prev_state is None:
            self.__prev_state = state_
        else:
            self.__update_td(self.__prev_state, state_, reward, terminal)
            self.__prev_state = state_

        # Treatment of terminal states
        if terminal:
            self.__set_terminal_value(state_)
            self.__prev_state = None

    def __add_new_state_if_unknown(self, state):
        """
        Add new state if the given state is new to the agent
        :param state:
        """
        if not self.__match(state):
            self.__values.update({state: self.__initial_v_val})

    def __set_terminal_value(self, state):
        """
        Set terminal value (zero) at terminal states
        :param state:
        """
        self.__values.update({state: 0.0})

    def _encode_state(self, state):
        if state in self._state_encoding_dict.keys():
            return self._state_encoding_dict[state]
        else:
            tag = len(self._state_encoding_dict.keys())
            self._state_encoding_dict.update({state : tag})
            return tag

    def __update_td(self, state, state_dash, reward, terminal):
        """
        Update value function by simple TD-learning algorithm
        :param state:
        :param state_dash:
        :param reward:
        :param terminal:
        """

        if terminal is False:
            td_error = float(reward) + self.__gamma*self.__values[state_dash] - self.__values[state]
        else:
            td_error = float(reward) - self.__values[state]

        new_value = self.__values[state] + self.__alpha * td_error
        self.__values.update({state: new_value})

    def __match(self, state):
        """
        Hard matching of the state
        :param state: some object
        :return: bool: True if state is matched in the history
        """
        return state in self.__values.keys()

    def get_v_value(self):
        """
        Obtain Value Function
        :return: value dictionary
        """
        return self.__values