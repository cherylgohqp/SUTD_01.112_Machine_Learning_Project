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
                data.append(lines[index:i])
                index = i + 1
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].split(' ')  # split the line into their respective parts

        # convert to keys and values (dict)
        for i in range(len(data)):
            # print(data[i])
            data_values = data[i]
            key = [word[0] for word in data_values]
            val = [word[1] for word in data_values]
            # print(key)
            # print(val)
            data[i] = [key, val]
            # dictionary = dict(zip(key,val))
            # print(data[i])
        # print(dictionary)

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                #print(data[i][0][j]) #give each word
                allKeys.append(data[i][0][j])
        setKeys = set(allKeys)

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                allVals.append(data[i][1][j])
        setVals = set(allVals)

    return dict(data=data, x_set=setKeys, y_set=setVals)


# Part 2a)
# e(x|y) = Count(y -> x)/Count(y)
# Count(y->x) means number of times you see x generated from y

def calculate_emission_count(parsed_data):
    data = parsed_data['data']
    x_set = parsed_data['x_set']
    y_set = parsed_data['y_set']
    # create a new datafram of zeros with keys as the index and sentiments as the columns
    count_emissions_df = pd.DataFrame(np.zeros((len(x_set), len(y_set))), index=x_set, columns=y_set)
    count_y = pd.Series(np.zeros(len(y_set)),
                        index=y_set)  # create a series object of zeros with index as the sentiments => to store the number times the sentiments appear
    # print(count_y)
    # print(count_emissions_df)

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
            count_y[sentiment] += 1
    return count_emissions_df, count_y


def get_emission_params(parsed_data):
    count_emissions_df, count_y = calculate_emission_count(parsed_data)
    return count_emissions_df / count_y  # e(x|y), where x is the tweet, and y is the sentiment


em_df = get_emission_params(obtain_data('SG/train'))
# get_emission_counts(obtain_data('sg_train'))
em_df.head()
