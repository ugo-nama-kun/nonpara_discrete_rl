# -*- coding: utf-8 -*-

from .context import nprl

import unittest
from nprl.q_learning import QLearning


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_init(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0:{0: 10, 1:100, 2:0}}
        qlean._state_action = (0, 0)

        qlean.init()

        self.assertEquals(qlean._q_values, {})
        self.assertIsNone(qlean._state_action)

    def test_get_max_q(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0:{0: 10, 1:100, 2:0}}

        max_val, best_action, action_and_q_dict = qlean._get_max_q(state=0)
        self.assertEquals(max_val, 100)
        self.assertEquals(best_action, 1)
        self.assertEquals(action_and_q_dict, {0:10, 1:100, 2:0})

    def test_get_action(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0:{0: 10, 1:100, 2:0}}

        qlean._epsilon = 0.0
        best_action = qlean._get_action(state=0, action_list=[0,1,2], test=False)
        self.assertEquals(best_action, 1)

        qlean._epsilon_test = 0.0
        best_action = qlean._get_action(state=0, action_list=[0, 1, 2], test=True)
        self.assertEquals(best_action, 1)

    def test_match_state(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 10, 1: 100, 2: 0}, 1:{0: 10, 1: 100, 2: 0}, 2:{0: 10, 1: 100, 2: 0}}

        self.assertEquals(qlean._match_state(state=0),
                          True)
        self.assertEquals(qlean._match_state(state=1),
                          True)
        self.assertEquals(qlean._match_state(state=2),
                          True)

        self.assertEquals(qlean._match_state(state=100),
                          False)

    def test_match_action(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 10, 1: 100, 2: 0}}

        for i in range(3):
            self.assertEquals(qlean._match_action(state=0, action=i), True)
        self.assertEquals(qlean._match_action(state=0, action=10), False)

    def test_add_new_state(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        self.assertEquals(qlean._q_values,{})

        qlean._add_new_state_action_if_unknown(state=0, action_list=[0,1,2])
        q_init = qlean._initial_q_val
        self.assertEquals(qlean._q_values,
                          {0: {0: q_init, 1: q_init, 2: q_init}})

        # Add action if action_list grew
        qlean._add_new_state_action_if_unknown(state=0, action_list=[0, 1, 2, 3])
        q_init = qlean._initial_q_val
        self.assertEquals(qlean._q_values,
                          {0: {0: q_init, 1: q_init, 2: q_init, 3: q_init}})

    def test_set_terminal_value(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 10, 1: 100, 2: 0}}

        qlean._set_terminal_value(state=0, action_list=[0,1,2])
        self.assertEquals(qlean._q_values,
                          {0: {0: 0.0, 1: 0.0, 2: 0.0}})

    def test_update_q(self):
        qlean = QLearning(lr=1.0, discount_factor=0.0, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 0}, 1: {0: 0}, 2: {0: 10}}

        qlean._update_q(state_action=(0,0),
                        next_state=1,
                        reward=1.0,
                        terminal=False)
        self.assertEqual(qlean._q_values,
                         {0: {0: 1.0}, 1: {0: 0}, 2:{0: 10}})

        qlean._update_q(state_action=(1, 0),
                        next_state=2,
                        reward=1.0,
                        terminal=True)
        self.assertEqual(qlean._q_values,
                         {0: {0: 1.0}, 1: {0: 1.0}, 2: {0: 10}})

    def test_learning_step(self):
        qlean = QLearning(lr=1.0, discount_factor=0.0, maximum_state_id=None, initial_fluctuation=False)
        q_init = qlean._initial_q_val

        # Initial Step
        qlean._add_new_state_action_if_unknown(state=0, action_list=[0,1,2])
        qlean._learning_step(new_state=0,
                             new_action=0,
                             reward=0.0,
                             terminal=False,
                             action_list=[0,1,2])
        self.assertEqual(qlean._q_values,
                         {0: {0: q_init, 1: q_init, 2:q_init}})
        self.assertEqual(qlean._state_action, (0,0))

        # Second Step
        qlean._add_new_state_action_if_unknown(state=1, action_list=[0,1,2])
        qlean._learning_step(new_state=1,
                             new_action=0,
                             reward=1.0,
                             terminal=False,
                             action_list=[0,1,2])
        self.assertEqual(qlean._q_values,
                         {0: {0: 1.0, 1: q_init, 2:q_init},
                          1: {0: q_init, 1: q_init, 2: q_init}})
        self.assertEqual(qlean._state_action, (1, 0))

        # Third Step : Terminal
        qlean._add_new_state_action_if_unknown(state=2, action_list=[0,1,2])
        qlean._learning_step(new_state=2,
                             new_action=0,
                             reward=1.0,
                             terminal=True,
                             action_list=[0, 1, 2])
        self.assertEqual(qlean._q_values,
                         {0: {0: 1.0, 1: q_init, 2: q_init},
                          1: {0: 1.0, 1: q_init, 2: q_init},
                          2: {0: 0.0, 1: q_init, 2: q_init}})

        self.assertEqual(qlean._state_action, None)

    def test_step(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 0, 1:-10}, 1: {0: 0}, 2: {0: 10}}

        self.assertEqual(qlean.step(new_state=0,
                                    reward=0,
                                    terminal=False,
                                    action_list=[0, 1],
                                    test=True),
                         0)

        self.assertEqual(qlean.step(new_state=1,
                                    reward=0,
                                    terminal=False,
                                    action_list=[0, 1],
                                    test=True),
                         0)

    def test_get_q_value(self):
        qlean = QLearning(discount_factor=0.5, maximum_state_id=None, initial_fluctuation=False)
        qlean._q_values = {0: {0: 0, 1: -10}, 1: {0: 0}, 2: {0: 10}}

        self.assertEqual(qlean._q_values,
                         {0: {0: 0, 1:-10}, 1: {0: 0}, 2: {0: 10}})


if __name__ == '__main__':
    unittest.main()