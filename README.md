
# Statistical Techniques Assignment 1

## Approximate Quantile Computation Algorithm (MRL98)

* [Dataset generation](main/dataset.py) - normal distribution rounded to the integers
* [MRL98](main/MRL98.py) implementation - based on the paper at http://www.cs.umd.edu/~samir/498/manku.pdf
* Some [tests](test.py)
* [Benchmarks](benchmarks.py) - [cProfiler](https://docs.python.org/3/library/profile.html) and [memory_profiler](https://pypi.org/project/memory-profiler/)
* [Plots](plots.ipynb)

## Memory and Time Complexities

Run memory profiling:

1) Install dependency

```shell
pip install memory_profiler
```

2.1) Powershell script

```shell
for ($i = 0; $i -le 4; $i++) {"$i"; python .\benchmarks.py $i}
```

2.2) Bash script

```shell
for i in 1 .. 4
do
  python .\benchmarks.py $i
done
```

3) To run time benchmarks, add "t" at the end of the python command:

```
python .\benchmarks.py $i t
```

