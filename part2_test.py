import unittest
import part2
import model


class Test2b(unittest.TestCase):
    def testSmoothedEmission(self):
        m = model.Model('SG/train')
        m.train()
        df = part2.GetEmissionDataFrame(m, 1)

        df2 = part2.GetEmissionDataFrame(m)
        print(part2.findMax(df2.iloc[18]))


