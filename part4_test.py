import unittest
import part4
from model import Model


class TestPart4(unittest.TestCase):
    def testSmall(self):
        m = Model('EN/train')
        m.train()
        perm = part4.GenerateFile(m)
        print(perm)

    def testGetDataframe(self):
        m = Model('EN/train')
        m.train()
        df = part4.GetTransitionDataFrame(m)
        print(df)




