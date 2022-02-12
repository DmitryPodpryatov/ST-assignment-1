import numpy

from dataset import generate_normal
from MRL98 import MRL98


def predict(stream, b, k, log):
    algorithm = MRL98(stream=stream, b=b, k=k, log=log)

    quantile = 0.75

    return algorithm(quantile)


def main():
    N = 10
    scale = 100

    b = 3
    k = 3

    epsilon = 1

    log = True
    stream = generate_normal(N=N, scale=scale, log=log)

    quantile = 0.75

    result = predict(stream, b, k, epsilon)
    print(f'Approx vs True: {result} - {numpy.quantile(stream, quantile)}')


if __name__ == '__main__':
    main()
