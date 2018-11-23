import pandas as pd
import numpy as np


def obtain_data(file):
    data = []
    dictionary = {}
    allKeys = []
    allVals = []

    with open(file, 'r', encoding="utf8") as f:
        lines = f.readlines()
        # print(len(lines))
        index = 0
        for i in range(len(lines) - 2000):
            # print(lines[i]) #eg. 'Omg O' is line[0]
            if lines[i] == '\n':
                data.append(lines[index:i])  # append everything until it encounters a \n
                index = i + 1
            lines[i] = lines[i].replace('\n', '')  # replace the \n at the end of each sentiments
            lines[i] = lines[i].split(' ')  # split the line into their respective parts
            # print(lines[i]) ##format = eg. ['Justin', 'B-neutral']
        # convert to keys and values (dict)
        for i in range(len(data)):
            # print(data[i])
            data_values = data[i]
            key = [word[0] for word in data_values]  # tweet
            val = [word[1] for word in data_values]  # sentiment
            # print(key)
            # print(val)
            data[i] = [key, val]
            # dictionary = dict(zip(key,val))
            # print(data[i])
        # print(dictionary)
        # data[i][0] gives the tweets
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                allVals.append(data[i][1][j])  # appending the sentiments corresponding to the tweets

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                # print(data[i][0][j]) #give each word
                allKeys.append(data[i][0][j])
        setKeys = set(allKeys)
        setVals = set(allVals)

    return dict(data=data, x_set=setKeys, y_set=setVals)


#obtain_data('SG\train')

# Part 2a)
# e(x|y) = Count(y -> x)/Count(y)
# Count(y->x) means number of times you see x generated from y

# e(x|y) = Count(y -> x)/Count(y)
# Count(y->x) means number of times you see x generated from y

import pandas as pd
import numpy as np


def calculate_emission_count(parsed_data):
    data = parsed_data['data']
    x_set = parsed_data['x_set']
    y_set = parsed_data['y_set']
    # create a new datafram of zeros with keys (ie.tweets) as the index and sentiments as the columns
    count_emissions_df = pd.DataFrame(np.zeros((len(x_set), len(y_set))), index=x_set, columns=y_set)
    count_y = pd.Series(np.zeros(len(y_set)),
                        index=y_set)  # create a series object of zeros with index as the sentiments => to store the number times the sentiments appear
    # print(count_y)
    # print(count_emissions_df) #datafram structure: where its tweets against columns of sentiments

    for word in data:
        # print(word) #format of data => [[keys],[values]]
        # keys are the tweets, values are the sentiments
        tweets_data, sentiments_data = word

        for i in range(len(tweets_data)):
            tweet, sentiment = tweets_data[i], sentiments_data[i]  # associate the tweet with its sentiment
            # print(tweet,sentiment)
            # print(sentiment)
            # +1 to the row,col, given the tweet, sentiment freq +1
            count_emissions_df.loc[tweet, sentiment] += 1  # .loc[] access a grp of rows and columns by labels
            # count_emissions_df is for Count(y->x) [counting the number of times a sentiment wrt to the tweet]
            count_y[sentiment] += 1  # incrementing the number of time the respective sentiment appear
    return count_emissions_df, count_y


def get_emission_params(parsed_data):
    count_emissions_df, count_y = calculate_emission_count(parsed_data)
    return count_emissions_df / count_y  # e(x|y), where x is the tweet, and y is the sentiment


em_df = get_emission_params(obtain_data('SG/train'))
# get_emission_counts(obtain_data('sg_train'))
em_df.head()


#Part 2b)

def calculate_new_emission_counts(parsed_data, k):
    count_emissions_df, count_y = calculate_emission_count(parsed_data)
    # .sum(axis = 1(sum the column), axis = 0 (sum the index))
    count_tweet_appearance = count_emissions_df.sum(axis=1)
    # print(count_tweet_appearance) #counting the number of times each tweet appears by summing everything across the columns
    '''Output of count_tweet appearance eg. 
        seems                      15.0
        https://t.co/h6Ie4IBJ08     1.0
        #AnnaVonHausswolff          2.0
        Bowery                      2.0
        refuge                      2.0
        @chuckielufc                1.0
        https://t.co/7xSNeWemp1     1.0
        @chris_steller              3.0
        unexpected                  3.0
        #usantdp                    1.0
        Ones                        2.0
        1979                        4.0
        @joceltsh                   1.0
        @TomBoxingAsylum            2.0
        @thistletat13               2.0
        @eibeibb                    2.0
        @TalatHussain12             1.0
        Ilkeston                    2.0
        @ricosua                    1.0
        Belarus                     2.0
        charms                      2.0
        @EvermorSolution            2.0
        https://t.co/WrcuWKQ0Xg     2.0
        FIRED                       2.0'''

    failed_tweets = count_tweet_appearance[count_tweet_appearance < k]
    # print(failed_tweets)
    '''eg output if k<3 (ie. tweets with occurence less than 3 times) is:
        @Nandos                    1.0
        @rcmpgrcpolice             1.0
        #yas                       1.0
        @just                      2.0
        ford                       1.0
        attracted                  2.0
        @Unitetheunion             1.0
        .....        '''

    # replace the tweets that occur less than 1.0 with "#UNK#"
    # print(failed_tweets.index) #gives all the tweets that <1.0

    replace_tweets = count_emissions_df.loc[failed_tweets.index].sum(axis=0)
    replace_tweets.name = '#UNK#'

    new_df = count_emissions_df.append(replace_tweets)
    new_df = new_df.drop(failed_tweets.index, axis=0)  # drop all failed_tweets words
    # print(new_df) #without failed_tweets words inside, has #UNK# row inside at the bottom

    return new_df, count_y


def get_new_emission_params(parsed_data, k):
    count_emissions_df, count_y = calculate_new_emission_counts(parsed_data, k)
    return count_emissions_df / count_y  # e(x|y), where x is the tweet, and y is the sentiment


# calculate_new_emission_counts(obtain_data('sg_train'),3)
new_em_df_parameters = get_new_emission_params(obtain_data('sg_train'), 1)
# new_em_df_parameters.sum(axis=1) #gives the sum of each rows (individual respective words)
'''eg.
Throwing          0.000012
headquarters      0.000300
insist            0.000012
except            0.000071
Broken            0.000128
LCD               0.000124
occur             0.000012
sound             0.000071
'''

new_em_df_parameters.sum(axis=0)  # gives the counts of the sentiments

#part 2c)
def training_dataset(file):
    dataset = obtain_data(file)
    k = 1
    return get_new_emission_params(dataset, k)


# single sentiment analysis for a word
def sentiment_analysis(emission_param, x):
    # checking if the tweet is an undiscovered/discovered word
    # if the word does not appear in training set, then change it to #UNK#
    # print(emission_param.index) #gives the individual tweets
    if x not in emission_param.index:
        x = '#UNK#'
    probability = emission_param.loc[x, :]
    # print(probability)
    max_probability = None
    for col in probability.index:
        # print(col) #gives the sentiments labels
        '''B-positive
            ..
            ...
            -
            I-negative
            B-neutral
            242
            O
            477
            B-negative
            .
            I-positive
            I-neutral'''
        if max_probability is None:
            max_probability = probability.loc[col]
            y = col
        elif probability.loc[col] > max_probability:  # take the max prob
            max_probability = probability.loc[col]
            y = col  # take the sentiment with the highest probability
    return y


def evaluation(filename, emission_param, outputfile):
    with open(filename, 'r', encoding="utf8") as inputfile:
        lines = inputfile.readlines()
        lines = [line.replace('\n', '') for line in lines]
        # print(lines)
        '''['best', 'friends', 'who', 'cry', 'on', 'FaceTime', 'together', ',', 'stay', 'together', '', "I'm", 'at', 'Starbucks',
        'in', 'Johor', 'Bahru', ',', 'Johor', 'w', '/', '@cassiecr17', 'https://t.co/3rzoTtjRag', '', 'Reports', 'of', 'a', 
        'collision', 'on', 'Friary', 'Road', 'in', 'Naas', 'https://t.co/MZgfLNdbyr', '', '♫', 'She', 'Moves', 'In', 'Her', 'Own' ......]
        '''

        for i in range(len(lines)):
            line = lines[i]  # each individual tweets
            if line != '':  # if line is not empty
                line = line + ' ' + sentiment_analysis(emission_param, line)
            line += '\n'
            lines[i] = line

        with open(outputfile, "w", encoding="utf8") as outputfile:
            for line in lines:
                outputfile.write(line)
    print("evaluation completed!")


