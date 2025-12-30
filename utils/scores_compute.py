import json
import os
from rouge import Rouge
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
# prediction file
prediction_file = os.path.join(project_root, 'results', 'test_predictions.jsonl')

def load_data(filename):
    """Load prediction file, extract predictions and reference labels"""
    preds = []
    refs = []
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return [], []

    print(f"Loading file: {filename} ...")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
                p = item.get('predict', '')
                r = item.get('label', '')

                # Filter out empty data to prevent errors
                if p and r:
                    # Key step: For Chinese evaluation, separate each character with a space
                    # Otherwise, rouge treats the entire sentence as one word
                    p_seg = ' '.join([char for char in p])
                    r_seg = ' '.join([char for char in r])
                    
                    preds.append(p_seg)
                    refs.append(r_seg)
            except:
                continue
    
    return preds, refs

def scores_compute():
    # load data
    preds, refs = load_data(prediction_file)
    
    if not preds:
        print("No valid data read. Please check if test_predictions.jsonl is generated successfully and contains the 'label' field.")
        return

    print(f"Valid samples: {len(preds)}")
    print("Calculating ROUGE scores...")

    # Initialize ROUGE calculator
    rouge = Rouge()
    
    # Calculate scores
    scores = rouge.get_scores(preds, refs, avg=True)

    # Write to txt file
    output_txt = os.path.join(project_root, 'results', 'rouge_scores.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("Evaluation Results (ROUGE Score)\n")
        f.write("="*40 + "\n")
        
        def write_metric(name, metrics):
            f.write(f"[{name}]:\n")
            f.write(f"  - Recall:    {metrics['r']*100:.2f}%\n")
            f.write(f"  - Precision: {metrics['p']*100:.2f}%\n")
            f.write(f"  - F1-Score:  {metrics['f']*100:.2f}%\n")
            f.write("-" * 20 + "\n")

        write_metric("ROUGE-1 (Unigram Overlap)", scores['rouge-1'])
        write_metric("ROUGE-2 (Bigram Overlap)", scores['rouge-2'])
        write_metric("ROUGE-L (Longest Common Subsequence)", scores['rouge-l'])
        f.write("="*40 + "\n")
        
        f.write("\nAnalysis Tips:\n")
        f.write("ROUGE-1: Measures information coverage; higher means better keyword capture.\n")
        f.write("ROUGE-2: Measures fluency; higher means more coherent phrases.\n")
        f.write("ROUGE-L: Measures sentence structure similarity.\n")
    print(f"ROUGE scores saved to: {output_txt}")

if __name__ == "__main__":
    scores_compute()