
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


obtain_data('SG/train')