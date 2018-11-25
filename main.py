from model import Model
import part2
import part3
import part4
import viterbi


if __name__ == "__main__":
    languages = ['SG', 'CN', "FR", "EN"]
    for lang in languages:
        m = Model(lang + "/train")
        m.train()

        # Emission dataframe from part 2
        em_df = part2.GetEmissionDataFrame(m, 1)

        # 1st order transition dataframe from part 3
        tr_df = part3.GetTransitionDataFrame(m)

        # 2nd order HMM transition dataframe from part 4
        tr_2nd_order = part4.GetTransitionDataFrame(m)

        readfile = lang + "/dev.in"

        # part 2 tagging
        part2.TagTweets(
            _out=lang+"/part2_out.txt",
            _emission_df=em_df,
            _file=readfile
        )

        # part 3 tagging
        viterbi.TagWithViterbi(
            _out=lang+"/part3_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df)

        # part 4 tagging
        viterbi.TagWithViterbi(
            _out=lang+"/part4_out.txt",
            _file=readfile,
            _model=m,
            _emission_df=em_df,
            _transition_df=tr_df,
            _2nd_order_df=tr_2nd_order)


