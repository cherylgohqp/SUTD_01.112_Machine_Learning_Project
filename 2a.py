import document


def createDocument(_data):
    d = document.Document(_data)
    d.parse()
    return d


def Count(d, _y, _x):
    try:
        y_count = d.y[_y]
        y_x_count = d.x[_x][_y]
        return y_count, y_x_count
    except:
        print("Exception")
        print(d.y)
        return 0, 1


# part a
# returns e(x|y) = Count(x -> y)/Count(y)
def emissionParameters(_document, _y, _x, _test=None):
    try:
        y_count = _document.y[_y]
        y_x_count = _document.x[_x][_y]
        return y_x_count / y_count
    except:
        print("Exception")
        print(_document.y)
        return 0/1


def train(_data):
    # trains using training data
    d = createDocument(_data)
    d.parse()
    return d


if __name__ == "__main__":
    d = train("SG/train")
    print(len(d.y))
    print(len(d.x))








