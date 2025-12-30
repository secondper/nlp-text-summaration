import json
import torch
from bert4torch.models import build_transformer_model

def get_bart_model(config_path, checkpoint_path=None, device='cuda'):
    """
    Encapsulate the construction logic of the BART model
    
    Args:
        config_path (str): Path to Hugging Face config.json
        checkpoint_path (str, optional): Path to pre-trained weights (.bin).
        device (str): 'cuda' or 'cpu'
    
    Returns:
        model: Constructed bert4torch model object
    """
    # Read configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        hf_config = json.load(f)

    # Parameter mapping (HuggingFace -> bert4torch)
    # Otherwise, "missing required positional arguments" error will occur
    bert4torch_args = {
        'model': 'bart', 
        'vocab_size': hf_config['vocab_size'],
        'hidden_size': hf_config['d_model'],
        'num_hidden_layers': hf_config['encoder_layers'],
        'num_attention_heads': hf_config['encoder_attention_heads'],
        'intermediate_size': hf_config['encoder_ffn_dim'],
        'hidden_act': hf_config['activation_function'],
        'dropout_rate': hf_config['dropout'],
        'max_position': hf_config['max_position_embeddings'],
        'segment_vocab_size': 0, 
    }

    # Build model
    # Note: config_path in build_transformer_model is set to None because we pass parameters manually via bert4torch_args
    model = build_transformer_model(
        config_path=None, 
        checkpoint_path=checkpoint_path,
        **bert4torch_args
    ).to(device)

    return model