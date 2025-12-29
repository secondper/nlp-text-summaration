import os
import json
import torch
from tqdm import tqdm
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder

# 配置参数
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
checkpoint_dir = os.path.join(project_root, 'checkpoint')
model_weights_dir = os.path.join(project_root, 'model_weights')

config_path = os.path.join(checkpoint_dir, 'config.json')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

# 测试数据路径
test_data_path = os.path.join(project_root, 'data', 'LCSTS_origin', 'my_test.jsonl')
# 结果保存路径
output_save_path = os.path.join(project_root, 'results', 'test_predictions.jsonl')

# 加载权重的路径
weight_path = os.path.join(model_weights_dir, 'bart_epoch_10.pt')

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
maxlen = 512        # 原文最大长度
max_target_len = 128 # 生成摘要最大长度

print(f"正在初始化... 设备: {device}")

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 构建模型结构
print("正在构建模型结构...")

# 读取 config.json
with open(config_path, 'r', encoding='utf-8') as f:
    hf_config = json.load(f)

# 参数映射
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

# 构建模型 (初始状态，权重是随机的)
model = build_transformer_model(
    config_path=None, 
    checkpoint_path=None, # 这里先不加载预训练权重，下面手动加载微调后的权重
    **bert4torch_args
).to(device)

# 加载训练好的权重
if os.path.exists(weight_path):
    print(f"正在加载微调权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # 切换到评估模式 (关闭 Dropout 等)
    print("权重加载成功")
else:
    print(f"找不到权重文件: {weight_path}")
    exit()

# 定义解码器
class ArticleSummaryDecoder(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states=None):
        token_ids = inputs[0] 
        # 调用模型预测
        logits = model.predict([token_ids, output_ids])
        return logits[-1][:, -1, :]

    def generate(self, text, topk=4):
        # 编码
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        
        # Beam Search 生成
        output_ids = self.beam_search([token_ids], top_k=topk)
        
        # 数据格式清洗    
        output_ids = output_ids[0]
            
        # 解码成文本
        return tokenizer.decode(output_ids)

# 初始化生成器
summary_generator = ArticleSummaryDecoder(
    bos_token_id=tokenizer._token_end_id,
    eos_token_id=tokenizer._token_end_id,
    max_length=max_target_len,
    device=device
)

# 推理函数
def predict_summary(text):
    """
    对外提供的接口函数
    """
    # 去掉换行符等
    text = text.replace('\n', ' ').replace('\r', '')
    summary = summary_generator.generate(text)
    return summary

if __name__ == "__main__":
    if not os.path.exists(test_data_path):
        print(f"错误：找不到测试文件 {test_data_path}")
        exit()

    print(f"读取文件: {test_data_path}")
    print(f"结果将保存至: {output_save_path}")

    # 计算总行数用于进度条
    total_lines = sum(1 for _ in open(test_data_path, 'r', encoding='utf-8'))
    
    results = []
    
    # 打开输入文件和输出文件
    with open(test_data_path, 'r', encoding='utf-8') as f_in, \
         open(output_save_path, 'w', encoding='utf-8') as f_out:
        
        # 使用 tqdm 显示进度条
        for line in tqdm(f_in, total=total_lines, desc="正在生成摘要"):
            line = line.strip()
            if not line: continue
            
            item = json.loads(line)
            # 获取原文
            text = item.get('text', '')
            ref_summary = item.get('summary', '')
            
            if not text: continue
            # 生成摘要
            generated_summary = predict_summary(text)
            
            # 构造结果对象
            result_item = {
                "text": text, # 原文
                "predict": generated_summary, # 你的模型生成的摘要
                "label": ref_summary # 参考摘要
            }
            # 实时写入文件
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            
    print("-" * 30)
    print("所有摘要生成完毕")
    print(f"已保存至文件: {output_save_path}")
    print("-" * 30)