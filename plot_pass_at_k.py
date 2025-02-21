import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from tqdm import tqdm
from run_name_mapping import run_name_map



def estimator(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def calculate_pass_rates(directory):
    pattern = os.path.join(directory, "question_*_sample_*.json")
    files = glob.glob(pattern)
    questions = {}
    for file in tqdm(files, desc=f"Processing files in {directory}"):
        with open(file, "r") as f:
            data = json.load(f)
        qid = data.get("question_id")
        if qid > 14:
            continue
        if qid not in questions:
            questions[qid] = []
        questions[qid].append(data)
    num_samples_list = [len(samples) for samples in questions.values()]
    num_correct_list = [sum(1 for s in samples if s.get("correctness")) for samples in questions.values()]
    overall_pass_rates = []
    for k in range(1, 9):
        rates = [estimator(n, c, k) for n, c in zip(num_samples_list, num_correct_list)]
        overall_pass_rates.append(np.mean(rates) if rates else 0)
    return list(range(1, 9)), overall_pass_rates

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot pass@k for scored samples from multiple directories")
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Directories containing scored JSON sample files")
    parser.add_argument("--output", default="pass_at_k.pdf", help="Output filename for the plot")
    args = parser.parse_args()
    plt.figure(figsize=(12, 10))
    for input_dir in args.input_dirs:
        k_values, pass_rates = calculate_pass_rates(input_dir)
        plt.plot(k_values, pass_rates, marker="o", linewidth=3, label=run_name_map[input_dir.replace("results/", "").replace("taco/", "").replace("aime24/", "").replace("taco_medium/", "")])
    plt.xlabel("k", fontsize=14)
    plt.ylabel("Pass@k", fontsize=14)
    plt.xticks(k_values, fontsize=12)
    plt.yticks(np.linspace(0, 0.6, 11), fontsize=12)
    plt.title("Pass@k", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    plt.show()

if __name__ == "__main__":
    main() 