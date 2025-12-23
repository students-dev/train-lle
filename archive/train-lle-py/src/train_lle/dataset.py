import csv
from .tensor import Tensor
import numpy as np

class Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    @staticmethod
    def from_csv(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
        data = np.array([[float(x) for x in row] for row in rows])
        inputs = data[:, :-1]
        targets = data[:, -1:]
        return Dataset(Tensor(inputs), Tensor(targets))