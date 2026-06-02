import ast
import statistics
import sys

from itertools import combinations
from pathlib import Path
from scipy.stats import wilcoxon

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

def load_results(file_path: str) -> tuple[dict[list[str]], list[str]]:
    data = {}
    methods = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue

            idx = line.rfind("]")
            values_str = line[:idx+1]
            method = line[idx+1:]

            values = ast.literal_eval(values_str)
            data[method] = values
            methods.append(method)

    methods.sort(reverse=True)
    return data, methods

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Incorrect number of arguments.")
        print("Use: python wilcoxon.py <result_path>")
        exit(1)

    result_path = Path(sys.argv[1])
    file_path   = result_path / "wilcoxon.txt"

    try:
        with open(file_path, "w") as file:
            pass
    except OSError as e:
        print(f"Error creating the file: {e}")
        exit(1)

    for instance in bks:
        print(f"\nDataset {instance}")

        with open(file_path, "a") as file:
            file.write(f"\nDataset: {instance}\n")
            file.close()

        results, methods = load_results(f"raw_results/{instance}.txt")

        for m1, m2 in combinations(methods, 2):
            A = results[m1]
            B = results[m2]

            _, p = wilcoxon(A, B)

            mean_A = statistics.mean(A)
            mean_B = statistics.mean(B)

            if p < 0.05:
                if mean_A > mean_B:
                    winner = m1
                else:
                    winner = m2
            else:
                winner = "tie"

            print(f"{m1:10} vs {m2:10} -> p={p:22}, winner={winner}")
            with open(file_path, "a") as file:
                file.write(f"{m1:10} vs {m2:10} -> p={p:22}, ")

                if winner == "tie":
                    file.write(f"tie.\n")
                else:
                    file.write(f"{winner}.\n")
                    
                file.close()