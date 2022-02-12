import itertools
import math
import numpy as np


class MRL98:
    def __init__(self, stream, b: int, k: int, log: bool = False):
        self.stream = stream
        self.N = len(stream)
        self.ptr = 0

        self.b = b
        self.k = k

        self.buffers = [[] for _ in range(self.b)]
        self.weights = [0 for _ in range(self.b)]
        self.levels = [0 for _ in range(self.b)]

        self.log = log

    def new(self, buffer_idx: int):
        lower_bound = buffer_idx * self.k
        upper_bound = lower_bound + self.k

        # Fill with -inf, +inf if there is not enough elements in the stream
        # (gotta make all buffers full or empty)
        if self.N < upper_bound:
            deficient = upper_bound - self.N
            print(f'.new {upper_bound, self.N, deficient}')

            self.stream += [-np.inf, np.inf] * (1 if deficient % 2 == 0 else 2)

            # Must add another buffer to balance the +- infs
            if deficient % 2 == 1:
                self.b += 1
                self.buffers += [[]]
                self.weights += [0]
                self.levels += [0]

        # Initialize the buffer and its weight
        self.buffers[buffer_idx] = self.stream[lower_bound: upper_bound]
        self.weights[buffer_idx] = 1

        if self.log:
            print('.new')
            self.log_()

    def collapse(self, *buffer_idxs):
        output = self.copy_and_merge(*buffer_idxs)

        # Put Y into one of the input buffers
        new_buffer_idx = min(buffer_idxs)

        # New weight is the sum of the weights
        self.weights[new_buffer_idx] = sum(self.weights[i] for i in buffer_idxs)

        w = self.weights[new_buffer_idx]

        if w % 2 == 0:
            offset = [w // 2, (w + 2) // 2]
        else:
            offset = [(w + 1) // 2]

        for i, ofst in zip(range(self.k), itertools.cycle(offset)):
            print(f'.collapse.offsets: {i, ofst}')
            self.buffers[new_buffer_idx][i] = output[i * w + ofst - 1]

        if self.log:
            print('.collapse')
            self.log_()

    def output(self, quantile):
        output = self.copy_and_merge(*range(self.b))

        W = sum(self.weights)

        return output[math.ceil(quantile * self.k * W)]

    def copy_and_merge(self, *buffer_idxs):
        # Make copies and sort the result of a merge
        output = []

        for idx in buffer_idxs:
            buffer, weight = self.buffers[idx], self.weights[idx]

            for item in buffer:
                output += [item] * weight

        if self.log:
            print(f'.cnm: {buffer_idxs, sorted(output)}')

        # Sort the result
        return sorted(output)

    def __call__(self, quantile):
        while [] in self.buffers:
            self.iteration()
        else:
            return self.output(quantile)

    def iteration(self):
        empty_idxs, full_idxs = [], []
        levels = []

        # Collect full and empty buffers plus the levels of full ones
        for i, buffer in enumerate(self.buffers):
            if buffer:
                full_idxs.append(i)
                levels.append(self.levels[i])
            else:
                empty_idxs.append(i)

        l = min(levels) if levels else -1

        if len(empty_idxs) == 0:
            level_l = [buffer_idx for buffer_idx in full_idxs if self.levels == l]

            self.collapse(level_l)
            self.levels[level_l[0]] = l + 1
        elif len(empty_idxs) == 1:
            i = empty_idxs[0]

            self.new(i)
            self.levels[i] = l
        else:
            for buffer_idx in empty_idxs:
                self.new(buffer_idx)
                self.levels[buffer_idx] = 0

    def log_(self):
        print(f'Buffers: {self.buffers}')
        print(f'Weights: {self.weights}')
        print(f'Levels: {self.levels}')
