import unittest
import part4
from model import Model


class TestPart4(unittest.TestCase):
    def testSmall(self):
        m = Model('SG/train')
        m.train()
        perm = part4.GenerateFile(m, False)
        print(perm)

    def testGetDataframe(self):
        m = Model('SG/train')
        m.train()
        df = part4.GetTransitionDataFrame(m)
        print(df)



