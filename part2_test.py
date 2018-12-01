import unittest
import part2
import model
import time


class Test2b(unittest.TestCase):
    def testSmoothedEmission(self):
        starttime = time.time()
        m = model.Model('SG/train')
        m.train()
        midtime = time.time()
        print("done training in {}s".format(midtime - starttime))
        df = part2.GetEmissionDataFrame(m, 1)
        mid2time = time.time()
        print("done getting emission dataframe in {}s".
              format(mid2time - midtime))

        part2.TagTweets('testdata/part2test.out', df, 'testdata/small_test')
        endtime = time.time()
        print("done with tagging in {}s".format(endtime - mid2time))


    def testCompleteSmoothed(self):
        m = model.Model('SG/train')
        m.train()
        df = part2.GetEmissionDataFrame(m, 1)

        print(df.shape)
        print(df)
        part2.TagTweets('SG/part2only.txt', df, 'SG/dev.in')


