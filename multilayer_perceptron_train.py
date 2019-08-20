import click
from math import isnan
import matplotlib.pyplot as plt
import numpy as np

from srcs.model import Model
from srcs.layers import Input, FC
from srcs.utils import read_data, to_categorical, normalise, str_to_int

@click.command()
@click.argument("data_file", default="data.csv", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    X, Y = read_data(data_file, sep, 1)

    # build model
    model = Model()
    out0 = model.add(Input(X.shape[1]))
    out1 = model.add(FC(out0, 8, "sigmoid"))
    out2 = model.add(FC(out1, 4, "sigmoid"))
    _ = model.add(FC(out2, 2, "softmax", is_last=True))
    model.show()

    # train / plot
    history = model.train(X, Y, batch_size=32, epochs=200, cross_validation=0.2, lr=0.1)
    plt.figure("Train history")
    plt.plot(history[:, 0], label="loss")
    plt.plot(history[:, 1], label="acc")
    plt.plot(history[:, 2], label="val_loss")
    plt.plot(history[:, 3], label="val_acc")
    plt.legend()
    plt.show()

    # save model
    model.save("model.json")

if __name__ == "__main__":
    main()