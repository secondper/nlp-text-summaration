import os
import sys
import json
import torch
from tqdm import tqdm
from bert4torch.tokenizers import Tokenizer

# 配置
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
sys.path.append(project_root)

from core.model import get_bart_model
from core.decoder import ArticleSummaryDecoder


# 资源路径
checkpoint_dir = os.path.join(project_root, 'checkpoint')
model_weights_dir = os.path.join(project_root, 'model_weights')

config_path = os.path.join(checkpoint_dir, 'config.json')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

# 测试数据路径
test_data_path = os.path.join(project_root, 'data', 'LCSTS_origin', 'my_test.jsonl')
# 结果保存路径
output_save_path = os.path.join(project_root, 'results', 'test_predictions.jsonl')

# 指定要使用的权重文件
weight_path = os.path.join(model_weights_dir, 'bart_epoch_10.pt')

# 超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
maxlen = 512         # 原文最大长度
max_target_len = 128 # 生成摘要最大长度

print(f"初始化推理脚本... 设备: {device}")

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 构建模型结构
print("正在构建模型结构...")
model = get_bart_model(
    config_path=config_path, 
    checkpoint_path=None, # 此时不加载预训练权重，稍后加载微调权重
    device=device
)

# 加载训练好的权重
if os.path.exists(weight_path):
    print(f"正在加载微调权重: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # 切换到评估模式
    print("权重加载成功")
else:
    print(f"错误: 找不到权重文件 {weight_path}")
    exit()

# 初始化解码器
summary_generator = ArticleSummaryDecoder(
    model=model,
    tokenizer=tokenizer,
    bos_token_id=tokenizer._token_end_id,
    eos_token_id=tokenizer._token_end_id,
    max_length=max_target_len,
    device=device
)


if __name__ == "__main__":
    # 检查输入文件
    if not os.path.exists(test_data_path):
        print(f"错误: 找不到测试数据文件 {test_data_path}")
        exit()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"读取文件: {test_data_path}")
    print(f"结果将保存至: {output_save_path}")

    # 计算总行数 (用于进度条)
    with open(test_data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 开始处理
    with open(test_data_path, 'r', encoding='utf-8') as f_in, \
         open(output_save_path, 'w', encoding='utf-8') as f_out:
        
        # 使用 tqdm 显示进度
        for line in tqdm(f_in, total=total_lines, desc="正在生成摘要"):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                text = item.get('text', '')
                ref_summary = item.get('summary', '') # 可能是空，如果只有 input
                
                if not text: continue
                
                # --- 核心生成步骤 ---
                # 直接调用封装好的 generate 方法
                generated_summary = summary_generator.generate(text, maxlen=maxlen, topk=4)
                
                # --- 构造结果 ---
                result_item = {
                    "text": text,
                    "predict": generated_summary,
                    "label": ref_summary
                }
                
                # 写入文件
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError:
                print(f"跳过格式错误的行: {line[:50]}...")
                continue
            except Exception as e:
                print(f"生成出错: {e}")
                continue

    print("-" * 30)
    print("所有摘要生成完毕！")
    print(f"结果已保存至: {output_save_path}")
    print("-" * 30)