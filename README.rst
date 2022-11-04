BenchOpt benchmark for Convolutional Sparse Coding
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of convolutional sparse coding:

$$
    \\min_{\\theta_1, \\ldots, \\theta_K in \\mathbb{R}^d} \\frac{1}{2} \\|y - \\sum_k=1^K d_k * \\theta_k\\|^2_2 + \\lambda \\sum_{k=1}^K \\|\\theta_k\\|_1
$$

where $K$ is the number of atoms in the dictionary, and $*$ denotes the convolution.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_csc
   $ benchopt run ./benchmark_csc

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_csc -s alphacsc -d simulated --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.


.. |Build Status| image:: https://github.com/benchopt/benchmark_csc/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_csc/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
