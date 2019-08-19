import click
from math import isnan
import numpy as np
from model import Model
from layers import Input, FC

def read_data(data_file, sep, labels_col_idx):
    with open(data_file, mode="r") as f:
        for j, line in enumerate(f):
            if j == 0:
                data = [[] for elem in line.split(sep)]
            for i, feat in enumerate(line.split(sep)):
                data[i].append(feat[:-1] if feat.endswith("\n") else feat)
    X = data[1:labels_col_idx] + data[labels_col_idx+1:]
    X = normalise(X)
    X = X.T
    Y = data[labels_col_idx]
    Y, nb_class = str_to_int(Y)
    Y = to_categorical(Y, nb_class)
    return X, Y

def to_categorical(Y, nb_class):
    values = range(nb_class)
    cat_Y = []
    for i, elem in enumerate(Y):
        if elem in values:
            cat_Y.append(np.zeros(nb_class))
            cat_Y[i][values.index(elem)] = 1
    return np.array(cat_Y)

def normalise(X):
    norm_X = []
    for x in X:
        x = np.array(x).astype(np.float)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        norm_X.append(x)
    return np.array(norm_X)

def str_to_int(Y):
    uniques = []
    for elem in Y:
        if elem not in uniques:
            uniques.append(elem)
    for i, elem in enumerate(Y):
        Y[i] = uniques.index(elem)
    return np.array(Y), len(uniques)

@click.command()
@click.argument("data_file", default="data.csv", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    X, Y = read_data(data_file, sep, 1)

    model = Model()
    out0 = model.add(Input(X.shape))
    out1 = model.add(FC(out0, 8, "sigmoid"))
    out2 = model.add(FC(out1, 4, "sigmoid"))
    out3 = model.add(FC(out2, 2, "sigmoid", is_last=True))

    model.show()

    model.train(X, Y, batch_size=32, epochs=200, lr=0.1)

if __name__ == "__main__":
    main()