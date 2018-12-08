import unittest
import part4
from model import Model


class TestPart4(unittest.TestCase):
    def testSmall(self):
        m = Model('EN/train')
        m.train()
        perm = part4.GetTransitionDataFrame(m)
        print(perm)




