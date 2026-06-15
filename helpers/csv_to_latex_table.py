import pandas as pd
import io

csv_data = """
Row,Avg_Perf_all_instances,Std_Perf_all_instances,Probability_of_good,Avg_Perf_selected_instances,Std_Perf_selected_instances,CV_model_accuracy,CV_model_precision,CV_model_recall,BoxConstraint,KernelScale
algo_a2c_mean_reward,2.352,1.471,0.91,2.917,1.576,48.6,98.3,44.3,2.742,14.359
algo_dqn_mean_reward,1.515,1.45,0.458,2.191,1.141,85.4,76.5,98.5,0.003,1.646
algo_ppo_mean_reward,2.464,1.383,0.792,3.082,1.652,66.7,91.2,64.0,6.47,8.298
Oracle,2.565,1.365,1.0,,,,,,,
Selector,2.468,1.374,0.847,2.468,1.374,,84.7,46.0,,
"""
LABEL = "tab:res-merge"

# Load data
df = pd.read_csv(io.StringIO(csv_data))

# Clean algorithm names (e.g., algo_a2c_mean_reward -> A2C)
def clean_name(name):
    if name.startswith('algo_') and name.endswith('_mean_reward'):
        return name.replace('algo_', '').replace('_mean_reward', '').upper()
    return name

df['Row'] = df['Row'].apply(clean_name)

# Sort rows to match the typical order (Algorithms first, then Selector, then Oracle)
cat_type = pd.CategoricalDtype(categories=['A2C', 'DQN', 'PPO', 'SAC', 'Selector', 'Oracle'], ordered=True)
df['Row_cat'] = df['Row'].astype(cat_type)
df = df.sort_values('Row_cat').drop('Row_cat', axis=1)

# Find the highest Probability_of_good (excluding Oracle) to bold it
max_prob = df[df['Row'] != 'Oracle']['Probability_of_good'].max()

# Generate LaTeX
latex_lines = [
    "\\begin{table}",
    "\\centering",
    "\\caption{Algorithm Performance Comparison.}",
    "\\label{" + LABEL + "}",
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
