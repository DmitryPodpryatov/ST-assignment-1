import cProfile
import sys

from memory_profiler import profile

from main.MRL98 import MRL98
from main.main import generate_normal


@profile
def predict_with_memory_profiling(algorithm, quantile):
    """
    More on memory profiling: https://pypi.org/project/memory-profiler/
    """
    return [algorithm(quantile)]


def predict(algorithm, quantile):
    return [algorithm(quantile)]


def benchmark(i):
    Ns = [100_000 * 10 ** n for n in range(5)]
    scale = 1000

    epsilon = 0.01
    quantile = 0.75

    # b and k follow from the table on page 6 of http://www.cs.umd.edu/~samir/498/manku.pdf
    # from the combination of N and epsilon
    bs = [7, 12, 6, 6, 6]
    ks = [217, 229, 472, 472, 472]

    output = []

    stream = generate_normal(N=Ns[i], scale=scale)
    algorithm = MRL98(stream=stream, b=bs[i], k=ks[i])

    if len(sys.argv) == 3 and sys.argv[2] == 't':
        # More on cProfile: https://docs.python.org/3/library/profile.html#module-profile
        with cProfile.Profile() as pr:
            predict(algorithm, quantile)

        pr.print_stats()
    else:
        predict_with_memory_profiling(algorithm, quantile)

    del stream
    del algorithm

    return output


if __name__ == '__main__':
    benchmark(int(sys.argv[1]))
