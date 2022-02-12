from numpy.random import normal


def generate_normal(N: int, scale: float, log: bool = False):
    dataset = normal(loc=scale / 2, scale=scale / 2, size=N)
    dataset = dataset.astype(int)

    if log:
        print(dataset)

    return dataset
