import unittest
import part2
import model


class Test2b(unittest.TestCase):
    def testSmoothedEmission(self):
        m = model.Model('EN/train')
        m.train()
        df = part2.GetEmissionDataFrame(m, 1)

        print(df.shape)
        print(df)

