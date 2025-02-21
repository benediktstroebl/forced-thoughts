import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from run_name_mapping import run_name_map

def get_token_length(text, model="gpt-4"):
    """Calculate number of tokens in text using tiktoken"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def analyze_results(task_name):
    # Find all results directories for the given task
    results_dir = Path("results") / task_name
    if not results_dir.exists():
        raise ValueError(f"No results found for task {task_name}")
        
    runs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    # Initialize data structures
    run_data = defaultdict(lambda: {
        'force_counts': Counter(),
        'output_lengths': [],
        'question_results': defaultdict(lambda: defaultdict(list))  # {question_id: {sample_idx: success}}
    })
    
    for run_dir in runs:
        run_name = run_dir.name
        
        # Process each result file in the run
        for result_file in run_dir.glob("question_*_sample_*.json"):
            with open(result_file) as f:
                data = json.load(f)
                
            # Count forces
            for force in data["applied_forces"]:
                if "approach_force" in force["name"]:
                    run_data[run_name]['force_counts']["approach_force"] += 1
                else:
                    run_data[run_name]['force_counts'][force["name"]] += 1
                    
            # Track output length using tiktoken
            run_data[run_name]['output_lengths'].append({
                'length': get_token_length(data["output"]),
                'question_id': data["question_id"]
            })
                
            # Track question results by sample
            question_id = data["question_id"]
            sample_idx = data["sample_index"]
            run_data[run_name]['question_results'][question_id][sample_idx].append(
                data.get("correctness", False)
            )

    # Create grid of plots for each run
    n_runs = len(runs)
    n_cols = 2
    n_rows = (n_runs + 1) // 2
    
    plt.figure(figsize=(15, 5*n_rows))
    
    for idx, (run_name, data) in enumerate(sorted(run_data.items())):
        display_name = run_name_map.get(run_name, run_name)
        
        # Force distribution
        plt.subplot(n_rows, n_cols*2, idx*2 + 1)
        force_df = pd.DataFrame.from_dict(data['force_counts'], orient='index', columns=['count'])
        sns.barplot(data=force_df, x=force_df.index, y='count')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{display_name}")
        
        # Output length distribution
        plt.subplot(n_rows, n_cols*2, idx*2 + 2)
        output_df = pd.DataFrame(data['output_lengths'])
        sns.histplot(data=output_df, x='length', bins=20)
        plt.title(f"{display_name}")
        
    plt.tight_layout()
    plt.savefig(f"results/{task_name}/run_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create success heatmap
    all_question_ids = sorted(set(qid for data in run_data.values() 
                                for qid in data['question_results'].keys()))
    run_names = sorted(run_data.keys())
    
    heatmap_data = []
    question_ids_plot = []
    for q_id in all_question_ids:
        if q_id > 50:
            break
        question_ids_plot.append(q_id)
        row = []
        for run in run_names:
            # Check if any sample for this question was successful
            samples_results = run_data[run]['question_results'].get(q_id, {})
            any_success = any(any(results) for results in samples_results.values())
            row.append(1 if any_success else 0)
        heatmap_data.append(row)
        
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, 
                xticklabels=[run_name_map.get(r, r) for r in run_names],
                yticklabels=question_ids_plot,
                cmap="YlOrRd",
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Success'})
    plt.xlabel("Run Configuration")
    plt.ylabel("Question ID")
    plt.title(f"{task_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"results/{task_name}/success_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    summary_stats = pd.DataFrame([{
        'Run': run_name_map.get(run, run),
        'Avg Tokens': np.mean([d['length'] for d in data['output_lengths']]),
        'Total Forces': sum(data['force_counts'].values()),
        'Success Rate': sum(1 for qid in data['question_results'] 
                          if any(any(results) for results in data['question_results'][qid].values())) / len(data['question_results']) * 100
    } for run, data in run_data.items()])
    
    summary_stats.to_csv(f"results/{task_name}/summary_stats.csv", index=False)

if __name__ == "__main__":
    # Example usage
    analyze_results("taco") 