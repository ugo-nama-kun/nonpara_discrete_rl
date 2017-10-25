# -*- coding: utf-8 -*-

from nprl.td import TDLearning
import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_run_td(self):

        td = TDLearning(gamma=0.5, maximum_state_id=None)

        # Simple one-way environment
        states = ['ONE', 'TWO', 'THREE', 'FOUR']

        for i in range(1000):
            now = i % len(states)
            if now == (len(states) - 1):
                reward = 1
                terminal = True
                td.step(states[now], reward, terminal)
            else:
                reward = 0
                terminal = False
                td.step(states[now], reward, terminal)

        # Check values
        self.assertAlmostEqual(td.get_v_value()[td._state_encoding_dict[states[3]]], 0, delta=0.1)
        self.assertAlmostEqual(td.get_v_value()[td._state_encoding_dict[states[2]]], 1.0, delta=0.05)
        self.assertAlmostEqual(td.get_v_value()[td._state_encoding_dict[states[1]]], 0.5, delta=0.05)
        self.assertAlmostEqual(td.get_v_value()[td._state_encoding_dict[states[0]]], 0.25, delta=0.05)


if __name__ == '__main__':
    unittest.main()