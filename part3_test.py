import unittest
import part3
import model
import part2helper
import viterbi
import time


class Test3(unittest.TestCase):
    def testSmoothedEmission(self):
        m = model.Model('SG/train')
        m.train()
        df = part3.GetTransitionDataFrame(m)
        print(df)

    def testSmall(self):
        starttime = time.time()
        m = model.Model("SG/train")
        m.train()
        traintime = time.time()
        print("Trained in {}s".format(traintime - starttime))

        readfile = "testdata/small_test"

        # Emission dataframe from part 2
        em_df = part2helper.GetEmissionDataFrame(m, 1)

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        dftime = time.time()
        print("DF generated in {}s".format(dftime - traintime))

        # part 3 tagging
        viterbi.TagWithViterbi(
            _out="testdata/output.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df)
        endtime = time.time()
        print("Tagged in {}s".format(endtime - dftime))

    def testEmission(self):
       m = model.Model('SG/train')
       m.train()
       df = part2helper.GetEmissionDataFrame(m)
       print('emission prob:\n\n', df.loc['not'])

       tr_df = part3.GetTransitionDataFrame(m)
       print('transition prob:\n\n', tr_df)

    def testDataFrame(self):
        m = model.Model('SG/train')
        m.train()
        tr_df = part3.GetTransitionDataFrame(m)
        print('transition prob:\n\n', tr_df)




