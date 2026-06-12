import pandas as pd
import io

csv_data = """
Row,Avg_Perf_all_instances,Std_Perf_all_instances,Probability_of_good,Avg_Perf_selected_instances,Std_Perf_selected_instances,CV_model_accuracy,CV_model_precision,CV_model_recall,BoxConstraint,KernelScale
algo_a2c_mean_reward,0.483,0.338,0.252,0.504,0.445,79.9,56.9,83.5,0.002,7.303
algo_ppo_mean_reward,0.456,0.378,0.207,0.506,0.589,85.0,60.2,81.5,0.003,1.646
algo_sac_mean_reward,0.727,0.375,0.86,0.878,0.242,86.0,97.1,86.3,0.039,0.502
Oracle,0.752,0.339,1.0,,,,,,,
Selector,0.697,0.447,0.873,0.708,0.446,,87.3,69.8,,
"""

# Load data
df = pd.read_csv(io.StringIO(csv_data))

# Clean algorithm names (e.g., algo_a2c_mean_reward -> A2C)
def clean_name(name):
    if name.startswith('algo_') and name.endswith('_mean_reward'):
        return name.replace('algo_', '').replace('_mean_reward', '').upper()
    return name

df['Row'] = df['Row'].apply(clean_name)

# Sort rows to match the typical order (Algorithms first, then Selector, then Oracle)
cat_type = pd.CategoricalDtype(categories=['A2C', 'PPO', 'SAC', 'Selector', 'Oracle'], ordered=True)
df['Row_cat'] = df['Row'].astype(cat_type)
df = df.sort_values('Row_cat').drop('Row_cat', axis=1)

# Find the highest Probability_of_good (excluding Oracle) to bold it
max_prob = df[df['Row'] != 'Oracle']['Probability_of_good'].max()

# Generate LaTeX
latex_lines = [
    "\\begin{table}",
    "\\centering",
    "\\caption{Algorithm Performance Comparison: Normalized Mean Reward.}",
    "\\label{tab:isa_perf}",
    "\\begin{tabular}{cccc}",
    "\\toprule",
    "\\textbf{Algorithms} & \\textbf{\\ Average performance\\ } & \\textbf{\\ Std performance\\ } & \\textbf{Probability of good} \\\\ \\midrule"
]

for _, row in df.iterrows():
    algo = row['Row']
    avg = f"{row['Avg_Perf_all_instances']:.3f}"
    std = f"{row['Std_Perf_all_instances']:.3f}"
    prob = float(row['Probability_of_good'])
    
    prob_str = f"{prob:.3f}"
    
    # Apply bolding for Selector and max probability
    if algo == 'Selector':
        algo = f"\\textbf{{{algo}}}"
        
    if prob == max_prob and row['Row'] != 'Oracle':
        prob_str = f"\\textbf{{{prob_str}}}"
        
    latex_lines.append(f"{algo} & {avg} & {std} & {prob_str} \\\\")

latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
])

# Print output
print("\n".join(latex_lines))
