import unittest
import part3
import model
import part2
import pandas as pd
import viterbi


class Test3(unittest.TestCase):
    def testSmoothedEmission(self):
       m = model.Model('SG/train')
       m.train()
       df = part3.GetTransitionDataFrame(m)

       # print(df.loc['__START__'])
       transition_prob = df.loc['__START__', 'O']
       print(transition_prob)
       print(df)

    def testDF(self):
       df = pd.DataFrame([[1, 2, 3], ['x', object, 'z']], index=['A', 'B'], columns=[0, 1, 2])
       df.loc['B', 1] = ['m', 'n', 'o', '', '', '', '']
       print(df)
       df.loc['B', 1] = ['m', 'n']
       print(df)

    def testSmall(self):
       m = model.Model("SG/train")
       m.train()

       readfile = "testdata/small_test"

       # Emission dataframe from part 2
       em_df = part2.GetEmissionDataFrame(m, 1)

       # 1st order transition dataframe from part 3
       tr_df = part3.GetTransitionDataFrame(m)

       # part 3 tagging
       viterbi.TagWithViterbi(
           _out="testdata/output.txt",
           _file=readfile,
           _model=m,
           _emission_df=em_df,
           _transition_df=tr_df)

    def testEmission(self):
       m = model.Model('SG/train')
       m.train()
       df = part2.GetEmissionDataFrame(m)
       print('emission prob:\n\n', df.loc['not'])

       tr_df = part3.GetTransitionDataFrame(m)
       print('transition prob:\n\n', tr_df)

    def testDataFrame(self):
        m = model.Model('SG/train')
        m.train()
        tr_df = part3.GetTransitionDataFrame(m)
        print('transition prob:\n\n', tr_df)




