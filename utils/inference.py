import os
import sys
import json
import torch
from tqdm import tqdm
from bert4torch.tokenizers import Tokenizer

# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
sys.path.append(project_root)

from core.model import get_bart_model
from core.decoder import ArticleSummaryDecoder


# source and destination paths
checkpoint_dir = os.path.join(project_root, 'checkpoint')
model_weights_dir = os.path.join(project_root, 'model_weights')

config_path = os.path.join(checkpoint_dir, 'config.json')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

# test dataset path
test_data_path = os.path.join(project_root, 'data', 'LCSTS_origin', 'my_test.jsonl')
# output save path
output_save_path = os.path.join(project_root, 'results', 'test_predictions.jsonl')

# specify the weight file to use
weight_path = os.path.join(model_weights_dir, 'bart_epoch_8.pt')

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
maxlen = 512         # max length of source text
max_target_len = 128 # max length of generated summary

print(f"Initializing inference script... Device: {device}")

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# build model structure
print("Building model structure...")
model = get_bart_model(
    config_path=config_path, 
    checkpoint_path=None, # Do not load pretrained weights here, will load fine-tuned weights later
    device=device
)

# Load fine-tuned weights
if os.path.exists(weight_path):
    print(f"正在加载微调权重: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # switch to evaluation mode
    print("Weights loaded successfully")
else:
    print(f"Error: Weight file not found {weight_path}")
    exit()

# Initialize decoder
summary_generator = ArticleSummaryDecoder(
    model=model,
    tokenizer=tokenizer,
    bos_token_id=tokenizer._token_end_id,
    eos_token_id=tokenizer._token_end_id,
    max_length=max_target_len,
    device=device
)


if __name__ == "__main__":
    # Check input file existence
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found {test_data_path}")
        exit()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading file: {test_data_path}")
    print(f"Results will be saved to: {output_save_path}")

    # Calculate total lines (for progress bar)
    with open(test_data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Start processing
    with open(test_data_path, 'r', encoding='utf-8') as f_in, \
         open(output_save_path, 'w', encoding='utf-8') as f_out:
        
        # Use tqdm to display progress
        for line in tqdm(f_in, total=total_lines, desc="Generating summaries"):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                text = item.get('text', '')
                ref_summary = item.get('summary', '') # May be empty if only input is provided
                
                if not text: continue
                
                # Core generation step
                # Directly call the encapsulated generate method
                generated_summary = summary_generator.generate(text, maxlen=maxlen, topk=4)
                
                # Construct result
                result_item = {
                    "text": text,
                    "predict": generated_summary,
                    "label": ref_summary
                }
                
                # Write to file
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line[:50]}...")
                continue
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

    print("-" * 30)
    print("All summaries generated!")
    print(f"Results saved to: {output_save_path}")
    print("-" * 30)