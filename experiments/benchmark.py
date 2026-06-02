import checker
import os
import statistics
import sys

from time import perf_counter

sys.path.append(os.path.abspath("../src"))
import main

bks = {
    "instance_0001": 15.0,
    "instance_0002": 2.0,
    "instance_0003": 12.0,
    "instance_0004": 3.5,
    "instance_0005": 177.88,
    "instance_0006": 691.0,
    "instance_0007": 392.25,
    "instance_0008": 162.94,
    "instance_0009": 4.42,
    "instance_0010": 17.11,
    "instance_0011": 16.85,
    "instance_0012": 11.25,
    "instance_0013": 117.38,
    "instance_0014": 181.64,
    "instance_0015": 149.33,
    "instance_0016": 85.0,
    "instance_0017": 36.5,
    "instance_0018": 117.2,
    "instance_0019": 202.0,
    "instance_0020": 5.0
}

methods = ["mbpso", "mbpsozt", "sbpso"]

def load_seeds(seed_path: str) -> list[int]:
    seeds = []

    try:
        with open(seed_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    seed = int(line)
                except ValueError:
                    print(f"Error: invalid format on the line {line_number}: '{line}'")
                    exit(1)
                
                seeds.append(seed)
        return seeds
    except FileNotFoundError:
        print(f"Error: file '{seed_path}' not found.")
        exit(1)
    except OSError as e:
        print(f"Error opening the file: {e}")
        exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Incorrect number of arguments.")
        print("Use: python benchmark.py <method_number> <seed_path>")
        print("Methods:\n\tMBPSO: 0\n\tMBPSO_zt: 1\n\tSBPSO: 2")
        exit(1)

    try:
        method = int(sys.argv[1])
    except ValueError:
        print("Error: method must be numeric integers.")
        exit(1)

    seeds_path = sys.argv[2]
    seeds = load_seeds(seeds_path)

    # Benchmark.
    print(f"\nMethod: {methods[method]}")
    
    with open(f"benchmarks/{methods[method]}.csv", "+w") as file:
        file.write(f"instance,min value,max value,mean values,stdev,gap,mean time\n")
        file.close()

    for instance, best_obj in bks.items():
        print(f"Dataset: {instance}")
        
        times = []
        objectives = []
        
        start = perf_counter()        
        for i in seeds:
            problem = main.main(f"../dataset/{instance}.txt", "./solution.txt", method, i)
            is_feasible, objective_value = checker.main(f"../dataset/{instance}.txt", "./solution.txt")

            if not is_feasible or problem.result["objective"] != objective_value:
                print(f"Something is wrong: {is_feasible}, {problem.result["objective"]}, {objective_value}")
                exit(1)
            
            objectives.append(problem.result["objective"])
            times.append(problem.result["time"])

        end = perf_counter()
        print(f"Time: {end - start}\n")

        min_obj = min(objectives)
        max_obj = max(objectives)
        mean_obj = statistics.mean(objectives)
        stdev_obj = statistics.stdev(objectives)
        gap = 100 * (best_obj - mean_obj) / best_obj

        mean_time = statistics.mean(times)

        with open(f"benchmarks/{methods[method]}.csv", "+a") as file:
            file.write(f"{instance},{min_obj},{max_obj},{mean_obj},{stdev_obj},{gap},{mean_time}\n")
            file.close()

        with open(f"raw_results/{instance}.txt", "+a") as file:
            file.write(str(objectives) + f"{methods[method]}\n")
            file.close()

    os.remove("./solution.txt")