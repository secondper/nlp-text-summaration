import os
import bert4torch

def fix_bert4torch_transformer():
    """
    automatically locate and fix the compatibility bug in transformers.py of bert4torch library
    """
    lib_path = os.path.dirname(bert4torch.__file__)
    target_file = os.path.join(lib_path, 'models', 'transformer.py')
    
    print(f"Locating file: {target_file}")
    
    if not os.path.exists(target_file):
        print("Error: transformer.py file not found!")
        return

    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and replace the buggy code line
    # Original: decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])
    # Fixed: decoder_outputs = self.decoder((decoder_input if isinstance(decoder_input, list) else [decoder_input]) + [encoder_hidden_states, encoder_attention_mask])
    
    bug_signature = "decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])"
    fixed_code = "decoder_outputs = self.decoder((decoder_input if isinstance(decoder_input, list) else [decoder_input]) + [encoder_hidden_states, encoder_attention_mask])"
    
    found = False
    for i, line in enumerate(lines):
        if bug_signature in line.strip():
            print(f"Found buggy code line (Line {i+1}): {line.strip()}")

            indent = line[:line.find(bug_signature.strip())] # Get original indentation
            lines[i] = indent + fixed_code.strip() + "\n"
            found = True
            break
   
    if found:
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("Fix successful! bert4torch library file has been updated.")
            print("You can now run train.py without issues.")
        except PermissionError:
            print("Permission denied! Please try running this script as an administrator.")
    else:
        print("Target code line not found, your bert4torch version might be newer or modified.No changes made.")

if __name__ == "__main__":
    fix_bert4torch_transformer()