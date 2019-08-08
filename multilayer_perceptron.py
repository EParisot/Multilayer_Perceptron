import click
import numpy as np
from model import Model

def read_data(data_file, sep, labels_col_idx):
    with open(data_file, mode="r") as f:
        for j, line in enumerate(f):
            if j == 0:
                data = [[] for elem in line.split(sep)]
            for i, feat in enumerate(line.split(sep)):
                data[i].append(feat[:-1] if feat.endswith("\n") else feat)
    X = data[:labels_col_idx] + data[labels_col_idx+1:]
    X = normalise(X)
    Y = data[labels_col_idx]
    Y = str_to_int(Y)
    return X, Y

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
    return np.array(Y)

@click.command()
@click.argument("data_file", default="data.csv", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
@click.option("-l", "labels_col_idx", default=0, help="labels column index")
def main(data_file, sep, labels_col_idx):
    X, Y = read_data(data_file, sep, labels_col_idx)
    model = Model()

if __name__ == "__main__":
    main()