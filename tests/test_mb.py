# -*- coding: utf-8 -*-

import copy
import numpy as np
import unittest

from nprl.model_based import ModelBased


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_init(self):
        mb = ModelBased(discount_factor=0.5,
                        exploration_rate=0.2,
                        maximum_state_id=200)

        mb._model = {"a": 112}
        mb._value_function = {"a": 123}
        mb._state_action = (1, 3)

        mb.init()

        self.assertEquals(mb._model, {})
        self.assertEquals(mb._value_function, {})
        self.assertIsNone(mb._state_action)

    def test_update_table_action(self):
        mb = ModelBased(discount_factor=0.5,
                        exploration_rate=0.2,
                        maximum_state_id=200)

        # Check initial entry
        state = 0
        action = 0
        action_list = [0, 1, 2]
        mb._update_table_state_action(state, action_list)
        self.assertEquals(mb._model[state].keys(), action_list)

        # Check the second entry with same action_list
        mb._update_table_state_action(state, action_list)
        self.assertEquals(mb._model[state].keys(), action_list)

        # Check the second entry with different action_list
        action_list.append(3)
        mb._update_table_state_action(state, action_list)
        self.assertEquals(mb._model[state].keys(), action_list)

    def test_update_transition_table(self):
        mb = ModelBased(discount_factor=0.5,
                        exploration_rate=0.2,
                        maximum_state_id=200)

        # Check initial entry
        state = 0
        action = 0
        action_list = [0, 1, 2]

        mb._update_table_state_action(state, action_list)

        # Check update of the transition table
        mb._update_transition_table(state,
                                    action,
                                    next_state=1,
                                    terminal=False)
        mb._update_transition_table(state,
                                    action,
                                    next_state=2,
                                    terminal=False)

        target = {1: {"__count": 0},
                  2: {"__count": 0},
                  "__count": 0,
                  "__reward": 0,
                  "__status": "unknown",
                  }
        self.assertEquals(mb._model[state][action],
                          target)

        # Check adding the Terminal State
        mb._update_transition_table(state,
                                    action,
                                    next_state=3,
                                    terminal=True)

        target = {1: {"__count": 0},
                  2: {"__count": 0},
                  3: {"__count": 0},
                  "__count": 0,
                  "__reward": 0,
                  "__status": "unknown",
                  }

        self.assertEquals(mb._model[state][action],
                          target)
        self.assertTrue(3 in mb._terminal_state_set)

    def test_model_update(self):
        mb = ModelBased(discount_factor=0.5,
                        exploration_rate=0.2,
                        maximum_state_id=200)

        # Initial Step
        state = "one"
        action_list = [0, 1, 2]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        # Check initial entry
        state = "one"
        action = 0
        reward = 1.0
        terminal = False
        next_state = "two"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        # Update the transition statistics
        print(mb._model)
        self.assertEquals(mb._model[state][action]["__count"], 1)
        self.assertEquals(mb._model[state][action][next_state]["__count"], 1)
        self.assertEquals(mb._model[state][action]["__reward"], reward)

    def test_approx_reward(self):
        """ Approximate reward: mean zero
        """
        mb = ModelBased()

        state = "one"
        action = 0
        terminal = True
        next_state = "two"
        action_list = [0]

        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)
        for i in range(10000):
            reward = np.random.normal(loc=0.0, scale=0.1)
            mb._update_model(state_action=(state, action),
                             next_state=next_state,
                             reward=reward,
                             terminal=terminal,
                             action_list=action_list)

        self.assertAlmostEquals(first=mb._model[state][action]["__reward"],
                                second=0.0,
                                delta=0.01)

    def test_greedy_action(self):
        mb = ModelBased()

        # Check initial entry

        # Initial Step
        state = "one"
        action_list = [0, 1, 2]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        state = "one"
        action = 0
        reward = 1.0
        terminal = True
        next_state = "two"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "one"
        action = 1
        reward = 3.0
        terminal = True
        next_state = "two"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "one"
        action = 2
        reward = 2.0
        terminal = True
        next_state = "two"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        print mb._model

        action, value = mb._get_greedy_action(state="one",
                                              action_list=action_list)

        self.assertEquals(action, 1)
        self.assertEquals(value, 3.0)

    def test_get_action_list(self):
        mb = ModelBased()

        # Check initial entry
        # Initial Step
        state = "one"
        action_list = [0, 1, 2]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        state = "one"
        action = 0
        reward = 1.0
        terminal = True
        next_state = "two"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        self.assertEquals(mb._get_action_set(state),
                          action_list)

    def test_value_iteration1(self):
        mb = ModelBased(discount_factor=0.0)

        # Initial Step
        state = "one"
        action_list = [0]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        state = "one"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "two"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "three"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 1.0
        terminal = False
        next_state = "one"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        mb._value_iteration()
        self.assertEquals(mb._value_function,
                          {"one": 0.0, "two": 0.0, "three": 1.0})

    def test_value_iteration2(self):
        # Value Iteration Test including the untried action
        mb = ModelBased(discount_factor=0.0, exploration_reward=10.0)

        # Initial Step
        state = "one"
        action_list = [0]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        state = "one"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "two"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "three"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "one"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        mb._value_iteration()
        self.assertEquals(mb._value_function,
                          {"one": 0.0, "two": 0.0, "three": 10.0})

    def test_value_iteration3(self):
        # Value Iteration Test including the terminal state
        mb = ModelBased(discount_factor=0.5)

        # Initial Step
        state = "one"
        action_list = [0]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        state = "one"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "two"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "three"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 1.0
        terminal = True
        next_state = "four"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "four"
        action = 0
        reward = 0.0
        terminal = True
        next_state = "four"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        mb._value_iteration()

        vf = {"one": 0.25, "two": 0.5, "three": 1.0, "four": 0.0}
        for key in vf.keys():
            self.assertAlmostEquals(mb._value_function[key], vf[key], delta=0.001)

    def test_value_iteration4(self):
        # Value Iteration Test in the three-state two-action MDP
        #
        # The exact value function is calculated by matrix with code:
        # ipython
        # : import numpy as np
        # : p0 = np.matrix([[0, 0, 1],[1,0,0],[0,1,0]])
        # : r0 = np.matrix([[-1], [-1], [+10]])
        # : gamma =0.5
        # : V0 = np.linalg.inv(np.eye(3) - gamma * p0)*r0
        # : print V0
        # output : [[  1.14285714][  4.28571429][ 10.57142857]]

        mb = ModelBased(discount_factor=0.5)

        # Initial Step
        state = "one"
        action_list = [0, 1]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)

        # Action 0
        state = "one"
        action = 0
        reward = -1.0
        terminal = False
        next_state = "two"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = -1.0
        terminal = False
        next_state = "three"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 10.0
        terminal = False
        next_state = "one"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        # Action 1
        state = "one"
        action = 1
        reward = -10
        terminal = False
        next_state = "three"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 1
        reward = +1.0
        terminal = False
        next_state = "two"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 1
        reward = +1.0
        terminal = False
        next_state = "one"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        mb._value_iteration()
        vf = {"one": 1.1428, "two": 4.2857, "three": 10.5714}
        for key in vf.keys():
            self.assertAlmostEquals(mb._value_function[key], vf[key], delta=0.01)

    def test_save_load_model(self):
        mb = ModelBased(discount_factor=0.0)

        # Initial Step
        state = "one"
        action_list = [0]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)


        state = "one"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "two"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = 0.0
        terminal = False
        next_state = "three"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 1.0
        terminal = False
        next_state = "one"
        action_list = [0]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        model_ = copy.deepcopy(mb._model)
        mb.save_model()

        mb._model = None
        mb.load_model()
        self.assertEquals(model_, mb._model)

    def test_plot_model(self):
        mb = ModelBased(discount_factor=0.5)

        # Initial Step
        state = "one"
        action_list = [0, 1]
        mb._update_model(state_action=None,
                         next_state=state,
                         reward=None,
                         terminal=False,
                         action_list=action_list)


        # Action 0
        state = "one"
        action = 0
        reward = -1.0
        terminal = False
        next_state = "two"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 0
        reward = -1.0
        terminal = False
        next_state = "three"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 50.0
        terminal = False
        next_state = "one"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        # Action 1
        state = "one"
        action = 1
        reward = -10
        terminal = False
        next_state = "three"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 1
        reward = +1.0
        terminal = False
        next_state = "two"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "three"
        action = 0
        reward = 0.0
        terminal = True
        next_state = "five"
        action_list = [0, 1]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        state = "two"
        action = 1
        reward = +3.0
        terminal = False
        next_state = "one"
        action_list = [0, 1, 2]

        mb._update_model(state_action=(state, action),
                         next_state=next_state,
                         reward=reward,
                         terminal=terminal,
                         action_list=action_list)

        print mb._model
        mb._value_iteration()

        print mb._value_function
        mb.show_model()


if __name__ == '__main__':
    unittest.main()
