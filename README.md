# Discrete PSO for Order Batching: A Comparative Study of Binary and Set-Based Solution Representations
This repository contains the source code for the algorithms used in the project, as well as the datasets used (provided by [Mercado Livre](https://github.com/mercadolibre/challenge-sbpo-2025/tree/master)).

The algorithms used were:
- Set-Based PSO (SBPSO) [[Langeveld and Engelbrecht, 2012]](https://www.researchgate.net/publication/257722514_Set-based_particle_swarm_optimization_applied_to_the_multidimensional_knapsack_problem)
- Modified Binary PSO (MBPSO) [[Yang et al., 2014]](https://ieeexplore.ieee.org/document/6661359)
- Modified Binary PSO with Zhang's transfer function (MBPSOzt) [[Zhang et al., 2020]](https://link.springer.com/chapter/10.1007/978-3-030-53956-6_18)

MBPSOzt was an adaptation, in which the transfer function of the original algorithm was replaced with one capable of controlling its opening and thus modifying the velocity-to-probability mapping.

## Experiments Design
The experiments were developed in Python 3.14.4 and run on an AMD Ryzen 5 7520U processor with 8 GB of memory, using Arch Linux. Runs were performed sequentially, rebooted between variants to reduce hardware interference and ensure consistent timing.

The 2025 SBPO challenge dataset consists of 20 instances. These instances vary in orders ($o$), item variety ($i$), ans aisles ($a$), as shown in the table below. Each method was executed 30 times per instance using fixed pseudorandom seeds in the range [10,000, 99,999], generated via the "True Random Number Generator" available at [Random.org](https://www.random.org/), totaling 1,800 runs.

> [!note]
> Used seeds: [10447, 22022, 24675, 35446, 35476, 37983, 39628, 40694, 41383, 42738, 45786, 46223, 48679, 56429, 56927, 58565, 60163, 61820, 61975, 65439, 68036, 68071, 72427, 73641, 78884, 82757, 85722, 87599, 94860, 95990].

| Dataset | Orders | Items | Aisles | Best-known solution (BKS) |
| ------- | ------ | ----- | ------ | ------------------------- |
| 01 | 61 | 155 | 116 | 15.00 |
| 02 | 7 | 7 | 33 | 2.00 |
| 03 | 82 | 246 | 124 | 12.00 |
| 04 | 16 | 59 | 91 | 3.50 |
| 05 | 2625 | 6407 | 161 | 177.88 |
| 06 | 10341 | 7089 | 184 | 691.00 |
| 07 | 8320 | 5747 | 180 | 392.25 |
| 08 | 2185 | 5831 | 168 | 162.94 |
| 09 | 70 | 222 | 304 | 4.42 |
| 10 | 1602 | 3689 | 383 | 17.11 |
| 11 | 1029 | 2784 | 375 | 16.85 |
| 12 | 133 | 337 | 342 | 11.25 |
| 13 | 8375 | 7525 | 413 | 117.38 |
| 14 | 12402 | 10974 | 413 | 181.64 |
| 15 | 7367 | 6633 | 402 | 149.33 |
| 16 | 1108 | 1051 | 88 | 85.00 |
| 17 | 417 | 411 | 83 | 36.50 |
| 18 | 2682 | 2309 | 90 | 117.20 |
| 19 | 2257 | 2104 | 134 | 202.00 |
| 20 | 5 | 5 | 5 | 5.00 |

The parameters used are all hard-coded in the `main.py` file.

## Repository structure
The project is organized into three main directories: `dataset`, `src`, and `experiments`.

The `dataset` directory contains the instances used in this work.

The `src` directory contains the implementation of the algorithms and all the methods developed. The `main.py` file is the entry point for executing the methods and takes the following parameters:
- the path to the instance;
- the path to the output file (in the format required by the SBPO 2025 challenge);
- the method identifier;
- the numerical seed.

The `methods` directory contains the implemented metaheuristics, while the `process` directory contains the code responsible for reading and modeling the instances into the format used by the methods.

The `experiments` directory contains the scripts responsible for running the benchmarks and statistical tests. The `benchmark.py` file takes as parameters the method identifier and a `.txt` file containing the seeds (one per line). The script then executes the specified method for all instances and seeds, saving:
- the aggregated results in a `.csv` file in the `benchmarks` directory;
- the individual results of each execution in the `raw_results` directory, which are later used in the statistical tests.

The `checker.py` file corresponds to the validator provided by the SBPO 2025 challenge. It is used by `benchmark.py` to verify that the generated solutions are valid. If an invalid solution is found, the execution is terminated.

Finally, the `wilcoxon.py` file reads the data stored in `raw_results` and performs the Wilcoxon statistical test for each instance, saving the results to the `wilcoxon.txt` file, which will be saved in the specified path.

Note that the scripts in the `experiments` directory are interdependent and, for this reason, use hardcoded paths, unlike the `src` directory, where paths are provided via parameters. Additionally, some method variations, such as different values of `k` in MBPSOzt, do not yet have automatic support for customizing the names of output files, requiring them to be renamed manually after execution.

## Execution
The project supports two types of execution:
1. Algorithm execution, which performs a single run of the selected method using the specified parameters;
2. Experiment execution, which performs multiple runs of a method using a set of seeds and stores the results obtained.

Before executing any component of the project, clone the repository:

```bash
git clone https://github.com/HenriUz/pso-for-order-batching.git
cd pso-for-order-batching
```

And if you use the mise:

```bash
mise trust
mise install
```

### Algorithms

```bash
cd src
python main.py <instance_path> <result_path> <method_number> <seed_number>
```

Available methods:
- `0`: MBPSO;
- `1`: MBPSOzt;
- `2`: SBPSO

### Experiments
The benchmark runs a method for all seeds provided in a file, saving the results of each execution. 

```bash
cd experiments
python benchmark.py <method_number> <seed_path>
```

### Statistical Test
The Wilcoxon test should be run after the benchmarks are complete, as it uses the results generated by them.

If the dependencies are not yet installed:
```bash
pip install -r experiments/requirements.txt
```

To run the test:
```bash
cd experiments
python wilcoxon.py <result_path>
```