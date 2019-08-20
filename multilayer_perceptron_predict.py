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

    model = Model().load_model("model.json")

if __name__ == "__main__":
    main()