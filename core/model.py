import json
import torch
from bert4torch.models import build_transformer_model

def get_bart_model(config_path, checkpoint_path=None, device='cuda'):
    """
    封装 BART 模型的构建逻辑
    
    Args:
        config_path (str): Hugging Face config.json 的路径
        checkpoint_path (str, optional): 预训练权重(.bin)的路径。
                                         如果为None，则随机初始化(用于加载微调后的权重前)。
        device (str): 'cuda' 或 'cpu'
    
    Returns:
        model: 构建好的 bert4torch 模型对象
    """
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        hf_config = json.load(f)

    # 参数映射 (HuggingFace -> bert4torch)
    # 否则会出现 "missing required positional arguments" 错误
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

    # 构建模型
    # 注意：build_transformer_model 的 config_path 设为 None，因为我们通过 args 手动传参了
    model = build_transformer_model(
        config_path=None, 
        checkpoint_path=checkpoint_path,
        **bert4torch_args
    ).to(device)

    return model