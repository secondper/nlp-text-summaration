import os
import json
import matplotlib.pyplot as plt
# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
result_file_path = os.path.join(project_root, 'results', 'results.jsonl')
save_dir = os.path.join(project_root, 'results', 'figs')
os.makedirs(save_dir, exist_ok=True)

# Read training results data
def read_results(file_path):
    experiments = []
    current_experiment = {
        'epochs': [],
        'losses': [],
        'rouge_1': [],
        'rouge_2': [],
        'rouge_l': []
    }
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return experiments

    try:
        last_epoch = -1
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    epoch = record['epoch']
                    
                    # If epoch resets (e.g. goes from 15 to 1), start a new experiment
                    if epoch <= last_epoch:
                        experiments.append(current_experiment)
                        current_experiment = {
                            'epochs': [],
                            'losses': [],
                            'rouge_1': [],
                            'rouge_2': [],
                            'rouge_l': []
                        }
                    
                    current_experiment['epochs'].append(epoch)
                    current_experiment['losses'].append(record['loss'])
                    current_experiment['rouge_1'].append(record['rouge-1']['f'] * 100)
                    current_experiment['rouge_2'].append(record['rouge-2']['f'] * 100)
                    current_experiment['rouge_l'].append(record['rouge-l']['f'] * 100)
                    
                    last_epoch = epoch
            
            # Add the last experiment
            if current_experiment['epochs']:
                experiments.append(current_experiment)
                
        print(f"Successfully read file: {file_path}")
        print(f"Found {len(experiments)} experiments.")
        return experiments
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

experiments = read_results(result_file_path)

if not experiments:
    print("No data found.")
    exit()

# Set font
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False   # Resolve minus sign display issue

# Create canvas (width 15, height 10)
plt.figure(figsize=(15, 10))

# Colors for different experiments
colors = ['#FF5733', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'd', 'v', '<']

# Chart 1: Training Loss Comparison
plt.subplot(2, 2, 1)
for i, exp in enumerate(experiments):
    label = f'Exp {i+1}'
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(exp['epochs'], exp['losses'], marker=marker, linestyle='-', color=color, label=label, linewidth=2)

plt.title('Training Loss Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Chart 2: ROUGE-1 Comparison
plt.subplot(2, 2, 2)
for i, exp in enumerate(experiments):
    label = f'Exp {i+1}'
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(exp['epochs'], exp['rouge_1'], marker=marker, linestyle='-', color=color, label=label)

plt.title('ROUGE-1 Score Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('ROUGE-1 (F1)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Chart 3: ROUGE-2 Comparison
plt.subplot(2, 2, 3)
for i, exp in enumerate(experiments):
    label = f'Exp {i+1}'
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(exp['epochs'], exp['rouge_2'], marker=marker, linestyle='-', color=color, label=label)

plt.title('ROUGE-2 Score Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('ROUGE-2 (F1)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Chart 4: ROUGE-L Comparison
plt.subplot(2, 2, 4)
for i, exp in enumerate(experiments):
    label = f'Exp {i+1}'
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(exp['epochs'], exp['rouge_l'], marker=marker, linestyle='-', color=color, label=label)

plt.title('ROUGE-L Score Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('ROUGE-L (F1)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Adjust layout
plt.tight_layout()

# Save and display
save_path = os.path.join(save_dir, 'experiment_comparison.png')
plt.savefig(save_path, dpi=300)
print(f"Chart generated: {save_path}")