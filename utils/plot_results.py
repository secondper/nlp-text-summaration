import matplotlib.pyplot as plt
import json
import os

# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
result_file_path = os.path.join(project_root, 'results', 'results.jsonl')
save_dir = os.path.join(project_root, 'results', 'figs')
os.makedirs(save_dir, exist_ok=True)

# Read training results data
def read_results(file_path):
    epochs = []
    losses = []
    rouge_1 = []
    rouge_2 = []
    rouge_l = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return epochs, losses, rouge_1, rouge_2, rouge_l
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    epochs.append(record['epoch'])
                    losses.append(record['loss'])
                    rouge_1.append(record['rouge-1']['f'] * 100)  # Convert to percentage
                    rouge_2.append(record['rouge-2']['f'] * 100)
                    rouge_l.append(record['rouge-l']['f'] * 100)
        print(f"Successfully read file: {file_path}")
        return epochs, losses, rouge_1, rouge_2, rouge_l
    except Exception as e:
        print(f"Error reading file: {e}")

epochs, losses, rouge_1, rouge_2, rouge_l = read_results(result_file_path)

# Set font
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False   # Resolve minus sign display issue

# Create canvas (width 12, height 5)
plt.figure(figsize=(12, 5))

# Chart 1: Training Loss Curve
plt.subplot(1, 2, 1) # 1 row 2 columns, 1st plot
plt.plot(epochs, losses, 'o-', color='#FF5733', label='Training Loss', linewidth=2)
plt.title('Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Chart 2: Validation ROUGE Scores Curve
plt.subplot(1, 2, 2) # 1 row 2 columns, 2nd plot
plt.plot(epochs, rouge_1, 's-', label='ROUGE-1', color='#1f77b4')
plt.plot(epochs, rouge_l, '^-', label='ROUGE-L', color='#2ca02c')
plt.plot(epochs, rouge_2, 'd-', label='ROUGE-2', color='#ff7f0e')

plt.title('Validation ROUGE Scores', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score (F1-Score)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Set Y-axis range
plt.ylim([min(min(rouge_1), min(rouge_2), min(rouge_l)) * 0.9, 
          max(max(rouge_1), max(rouge_2), max(rouge_l)) * 1.1])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save and display
save_path = os.path.join(save_dir, 'training_analysis.png')
plt.savefig(save_path, dpi=300) # Save high-resolution image
print(f"Chart generated: {save_path}")