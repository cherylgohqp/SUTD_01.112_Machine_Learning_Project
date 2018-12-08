from model import Model
import part2
import part3
import part4
import viterbi
import time


if __name__ == "__main__":
    begintime = time.time()
    languages = ["EN", 'SG', 'CN', "FR"]
    for lang in languages:
        print("Starting language {}".format(lang))
        starttime = time.time()
        m = Model(lang + "/train")
        m.train()
        traintime = time.time()
        print("Finished training {} in {}s".format(lang, starttime-traintime))

        # Emission dataframe from part 2
        em_df = part2.GetEmissionDataFrame(m, 1)

        part2time = time.time()
        print("Finished part2 df in {}s".format(part2time - traintime))

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        part3time = time.time()
        print("Finished part3 df in {}s".format(part3time - part2time))

        # 2nd order HMM transition dataframe from part 4
        tr_2nd_order = part4.GetTransitionDataFrame(m)

        part4time = time.time()
        print("Finished part4 df in {}s".format(part4time - part3time))

        print("    ---- ---- ----    \n")

        readfile = lang + "/dev.in"

        # part 2 tagging
        part2.TagTweets(
            _out=lang+"/dev.part2.out",
            _emission_df=em_df,
            _file=readfile
        )

        part2tagging = time.time()
        print("Finished tagging part2 in {}s".format(part2tagging - part4time))

        # part 3 tagging
        viterbi.TagWithViterbi(
            _out=lang+"/dev.part3.out",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df)

        part3tagging = time.time()
        print("Finished tagging part3 in {}s".format(part3tagging - part2tagging))

        # part 4 tagging
        viterbi.TagWithViterbi(
            _out=lang+"/dev.part4.out",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
            _2nd_order_df=tr_2nd_order)

        part4tagging = time.time()
        print("Finished tagging part4 in {}s".format(part4tagging - part3tagging))

        print("====================\n")

    endtime = time.time()
    print("Finished everything in {}s".format(endtime - begintime))
