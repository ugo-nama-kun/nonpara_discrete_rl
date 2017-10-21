# -*- coding: utf-8 -*-
"""
Python implementation of a simple Model-based RL for discrete-finite MDPs
Naoto Yoshida

This implementation is based on two components:
- Non-Parametric Dynamics Model Learning (Table)
- Value Iteration
"""

import copy
import numpy as np
import random

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


class ModelBased(object):
    NULL_ACTION = 0

    def __init__(self,
                 discount_factor=0.95,
                 exploration_rate=0.2,
                 exploration_rate_test=0.05,
                 exploration_reward=1.0,
                 vi_iteration=100,
                 vi_error_limit=0.01,
                 maximum_state_id=10000):
        """ Model-based RL Learning Agent

        :param float discount_factor: discount factor [0, 1)
        :param float exploration_rate: exploration rate in the e-greedy policy
        :param int maximum_state_id: maximum number of states that the agent generates
        """

        # Model Description
        # {
        #   state:
        #       {action:
        #           {next_state:
        #               {"__count": n_sas},
        #           "__count": n_sa,
        #           "__reward": reward estimation
        #           }
        #       }
        #   }
        # }

        self._model = {}
        self._value_function = {}  # {state: value}
        self._terminal_state_set = set()

        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_rate_test = exploration_rate_test
        self._initial_value = 0.0
        self._exploration_reward = exploration_reward
        self._maximum_state_id = maximum_state_id

        # Parameters of Value Iteration
        self._vi_error_limit = vi_error_limit
        self._vi_max_iteration = vi_iteration

        # Initialization of parameters used in the algorithm
        self._state_action = None

    def init(self):
        """ Initialize agent to the initial status

        :return:
        """
        self._model = {}
        self._value_function = {}
        self._terminal_state_set = set()
        self._state_action = None

    def reset(self):
        """ Reset agent to the initial status for new episode

        :return:
        """
        self.init()

    def _match_state(self, state):
        """ Hard matching of the state.

        :param state:
        :return: bool: True if state is matched in the history
        """
        return state in self._model.keys()

    def _match_action(self, state, action):
        """ Hard matching of the action given a state.

        :param state:
        :param action:
        :return: bool: True if state is matched in the history
        """
        return action in self._model[state].keys()

    def _update_model(self,
                      state_action,
                      next_state,
                      reward,
                      terminal,
                      action_list):
        """ Update the table model of the environment

        :param tuple state_action: state-action tuple of the previous state-action pair
        :param next_state: the consequence of the action at the state
        :param float reward: the reward wrt the transition
        :param bool terminal: terminal flag if the next_state is a terminal state or not
        :param list action_list:  action set at the state: A(s)
        :return:
        """

        state = state_action[0]
        action = state_action[1]

        # Update Table
        self._update_table(state, action, next_state, action_list, terminal)

        # Update the transition statistics
        self._model[state][action]["__count"] += 1
        self._model[state][action][next_state]["__count"] += 1
        # Set the (s, a) to the "known" status.
        if self._model[state][action]["__status"] == "unknown":
            self._model[state][action]["__status"] = "known"

        # MLE Update of the reward statistics
        alpha = 1.0 / float(self._model[state][action]["__count"])
        reward_prev = self._model[state][action]["__reward"]
        self._model[state][action]["__reward"] = alpha * reward + (1.0 - alpha) * reward_prev

    def _update_table(self, state, action, next_state, action_list, terminal):
        """ Update the form of the table model

        :param state:
        :param action:
        :param next_state:
        :param action_list:
        :param terminal:
        :return:
        """

        # Update and grow the model table if necessary. But the added tables are "unknown" status
        self._update_table_state_action(state, action_list)

        # Update the transition model given the table model. The "unknown" state only can change to "known" in this process
        self._update_transition_table(state, action, next_state, terminal)

    def _update_table_state_action(self, state, action_list):
        """ Update the table of the model.

        :param state:
        :param action_list:
        :return:
        """
        if self._match_state(state):
            # if state is already experienced, but action set may be changed
            for a in action_list:
                if not a in self._model[state].keys():
                    self._add_new_action(state, a)
        else:
            # if state have never seen experienced until now
            self._add_new_state(state)
            for a in action_list:
                self._add_new_action(state, a)

    def _add_new_state(self, state):
        """ Add a new state in model and the value function table

        :param state:
        :param terminal:
        :return:
        """
        self._model.update({state: {}})
        self._value_function.update({state: self._initial_value})

    def _add_new_action(self, state, action):
        """ Add a new action in the model table

        :param state:
        :param action:
        :return:
        """
        self._model[state].update({action:
                                       {"__count": 0,
                                        "__reward": 0,
                                        "__status": "unknown"}
                                   })

    def _update_transition_table(self, state, action, next_state, terminal):
        """ Update and grow the model table. NOTE: terminal signal is associated with next_state!

        :param state:
        :param action:
        :param next_state:
        :param bool terminal: if the next_state is a terminal state or not
        :return:
        """

        if terminal and not (next_state in self._terminal_state_set):
            # next_state is the terminal state. this is a special state
            self._model[state][action].update({next_state: {"__count": 0}})

            # Add next state in the terminal set
            self._terminal_state_set.add(next_state)
            return

        if not next_state in self._model[state][action].keys():
            # Add next state subsequent to the state-action
            self._model[state][action].update({next_state: {"__count": 0}})

        if not next_state in self._model.keys():
            # Add next state is it is unknown state
            self._add_new_state(next_state)

            # Check whether the state space is smaller than the limit
            assert len(self._model.keys()) < self._maximum_state_id, "Too much state. The number of states exceed the limit. : |S| > {}".format(self._maximum_state_id)

    def _get_action(self, state, action_list, test=False):
        """ Obtain next action with respect to the given state

        :param state: current state
        :param action_list: action candidates at the given state
        :param bool test: switch the agent take test behavior or learning behavior
        :return:
        """

        if test:
            if random.random() < self._exploration_rate_test:
                return random.choice(action_list)
            else:
                action, _ = self._get_greedy_action(state, action_list)
                return action
        else:
            # In training phase, epsilon-greedy
            if random.random() < self._exploration_rate:
                return random.choice(action_list)
            else:
                action, _ = self._get_greedy_action(state, action_list)
                return action

    def _get_greedy_action(self, state, action_list):
        """ Obtain a greedy action wrt the current model on the given state and an action list

        :param state:
        :param action_list:
        :return:
        """
        # In test phase, get greedy action from VF and model

        # If the given state is known as a terminal state
        if state in self._terminal_state_set:
            best_action = self.NULL_ACTION
            best_value = 0.0
            return best_action, best_value

        best_value = -np.inf
        best_action = None
        for a in action_list:
            # Calculate value
            if self._model[state][a]["__status"] == "known":
                r = self._model[state][a]["__reward"]
                v_ = 0.0
                for next_state in self._get_next_state_list(state, a):
                    p_sas = self._model[state][a][next_state]["__count"] / self._model[state][a]["__count"]
                    v_ += p_sas * self._value_function[next_state]
                value = r + self._discount_factor * v_
            else:
                # Set a default value as an exploration parameter if (s,a) set have never been executed
                value = self._exploration_reward

            # Compare
            if best_value < value:
                best_action = a
                best_value = value
        return best_action, best_value

    def _get_next_state_list(self, state, action):
        """ Obtain the next_state list given a state and an action

        :param state:
        :param action:
        :return:
        """
        next_state_list = []
        for c in self._model[state][action].keys():
            if c != "__count" and c != "__reward" and c != "__status":
                next_state_list.append(c)
        return next_state_list

    def _get_action_set(self, state):
        """
        Get action set given a state
        :param state:
        :return:
        """
        return self._model[state].keys()

    def _value_iteration(self):
        """ Value Iteration

        """
        # Naive implementation of value iteration
        for n in range(self._vi_max_iteration):
            max_error = 0.0
            for state in self._value_function.keys():
                v = copy.deepcopy(self._value_function[state])
                _, max_v = self._get_greedy_action(state=state,
                                                   action_list=self._get_action_set(state))
                self._value_function[state] = copy.deepcopy(max_v)
                max_error = np.max([max_error, np.abs(v - max_v)])
            if max_error < self._vi_error_limit:
                break

    def step(self, new_state, reward, terminal, action_list, test=False):
        """ Take a single step of the agent

        :param new_state: state observed by agent
        :param float reward: reward as a result of (state, action, new_state) triple
        :param bool terminal: Terminal flag
        :param list action_list: action set at new_state A(s)
        :param bool test:
        :return:
        """
        assert len(
            action_list) > 0, "action_list has to have at least one action"

        # Expand dictionary if new_state is unknown
        if not test:
            if self._state_action is not None:
                # Model Updating
                self._update_model(self._state_action,
                                   new_state,
                                   reward,
                                   terminal,
                                   action_list)

                # Value Function Update
                self._value_iteration()

        # Take an action
        new_action = self._get_action(new_state, action_list, test)

        self._state_action = (new_state, new_action)
        return new_action

    def get_model(self):
        """ Obtain Environment Model

        :return: environment model dictionary
        """
        return self._model

    def save_model(self):
        """ Save model as "model.json" at the current directory

        :return:
        """
        import pickle

        with open("model.pickle", "w") as handle:
            logger.info("Model Saving..b.")
            pickle.dump(self._model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Model Saved.")

    def load_model(self):
        """ Load model from "model.json" from the current directory

        :return:
        """
        import pickle

        with open("model.pickle", "rb") as handle:
            logger.info("Model Loading...")
            self._model = pickle.load(handle)
            logger.info("Model Loaded.")

    def show_model(self):
        """ Visualize the current environment model

        :return:
        """

