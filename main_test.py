import unittest
import part3
import model
import part2helper
import pandas as pd
import viterbi
import part4
import numpy as np


class TestFullMethods(unittest.TestCase):
    def testPart2WithTestdata(self):
        m = model.Model("SG/train")
        m.train()

        readfile = "testdata/small_test"

        # Emission dataframe from part 2
        em_df = part2helper.GetEmissionDataFrame(m, 1)

        part2helper.TagTweets(
            _out='testdata/part2_out.txt',
            _emission_df=em_df,
            _file=readfile
        )

    def testPart3WithTestdata(self):
        m = model.Model("SG/train")
        m.train()

        readfile = "testdata/small_test"

        # Emission dataframe from part 2
        em_df = part2helper.GetEmissionDataFrame(m, 1)

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        viterbi.TagWithViterbi(
            _out="testdata/part3_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
        )


    def testPart4WithTestdata(self):
        m = model.Model("SG/train")
        m.train()

        readfile = "testdata/small_test"

        # Emission dataframe from part 2
        em_df = part2helper.GetEmissionDataFrame(m, 1)

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        # 2nd order transition dataframe from part 4
        tr_2_df = part4.GetTransitionDataFrame(m)

        # print(tr_2_df)
        # quit()

        # part 4 tagging
        viterbi.TagWithViterbi(
            _out="testdata/part4_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
            _2nd_order_df=tr_2_df,
        )

    def testAll(self):
        m = model.Model("EN/train")
        m.train()

        readfile = "testdata/small_test"

        # Emission dataframe from part 2
        em_df = part2helper.GetEmissionDataFrame(m, 1)

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        # 2nd order transition dataframe from part 4
        tr_2_df = part4.GetTransitionDataFrame(m)

        # part 2 tagging
        part2helper.TagTweets(
            _out='testdata/part2_out.txt',
            _emission_df=em_df,
            _file=readfile
        )

        # part 3 tagging
        print("part 3:\n")
        viterbi.TagWithViterbi(
            _out="testdata/part3_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
        )

        print("part 4:\n")
        # part 4 tagging
        viterbi.TagWithViterbi(
            _out="testdata/part4_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
            _2nd_order_df=tr_2_df,
        )
