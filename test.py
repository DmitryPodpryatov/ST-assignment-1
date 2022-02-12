import numpy as np
from main.MRL98 import MRL98

def test_new():
    test_stream = [1, 2, 3, 4, 5]

    algorithm = MRL98(stream=test_stream, b=2, k=3, log=True)

    algorithm.new(0)
    algorithm.new(1)
    algorithm.new(2)

    assert algorithm.buffers == [[1, 2, 3], [4, 5, -np.inf], [np.inf, -np.inf, np.inf]]


def test_copy_and_merge():
    test_stream = [1, 2, 3, 4, 5]
    weights = [2, 2]

    algorithm = MRL98(stream=test_stream, b=2, k=2, log=True)

    algorithm.new(0)
    algorithm.new(1)

    algorithm.weights = weights
    output = algorithm.copy_and_merge(0, 1)

    assert output == [1, 1, 2, 2, 3, 3, 4, 4]


def test_collapse_on_example_from_paper():
    input_stream = [
        12, 52, 72, 102, 132,
        23, 33, 83, 143, 153,
        44, 64, 94, 114, 124
    ]

    weights = [2, 3, 4]

    algorithm = MRL98(stream=input_stream, b=3, k=5, log=True)

    for i in range(3):
        algorithm.new(i)

    algorithm.weights = weights
    algorithm.collapse(0, 1, 2)

    assert algorithm.buffers[0] == [23, 52, 83, 114, 143]
