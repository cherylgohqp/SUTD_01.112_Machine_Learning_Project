import unittest
import part2
import model
import time


class Test2b(unittest.TestCase):

    def testSimpleDF(self):
        m = model.Model('SG/train')
        m.train()
        df = part2.GetEmissionDataFrame(m, 1)
        print(df)


    def testCompleteSmoothed(self):
        starttime = time.time()
        m = model.Model('SG/train')
        m.train()

        midtime = time.time()
        print("Trained in {}s".format(midtime-starttime))
        df = part2.GetEmissionDataFrame(m, 1)
        endtime = time.time()
        print("Finished in {}s".format(endtime-starttime))
        # print("Final emission:", df)

        part2.TagTweets('testdata/part2test.out', df, 'testdata/small_test')
        endtime2 = time.time()
        print("done with tagging in {}s".format(endtime2 - endtime))

        # print(df.shape)
        # print(df)
        # part2.TagTweets('SG/part2only.txt', df, 'SG/dev.in')


