import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.generation import AutoRegressiveDecoder
from tqdm import tqdm
import os
from rouge import Rouge
from transformers import get_linear_schedule_with_warmup

# 配置参数
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

checkpoint_dir = os.path.join(project_root, 'checkpoint')
config_path = os.path.join(checkpoint_dir, 'config.json')
checkpoint_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

train_data_path = os.path.join(project_root, 'data', 'LCSTS_origin', 'train.jsonl')
valid_data_path = os.path.join(project_root, 'data', 'LCSTS_origin', 'valid.jsonl')

# 超参数
maxlen = 512 # 原文最大长度
max_target_len = 128 # 摘要最大长度
batch_size = 32
epochs = 15
learning_rate = 5e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置随机种子
seed_everything(42)

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 数据处理
class SummaryDataset(Dataset):
    """
    文本摘要数据集
    读取 jsonl 格式的数据，每行是一个 json 对象，包含 'text' 和 'summary' 字段
    训练集只取 20000 条，验证集只取 2000 条
    """
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                # 训练集只取 20000 条
                if "train.jsonl" in file_path and len(self.data) > 20000: break 
                # 验证集只取 2000 条
                if "valid.jsonl" in file_path and len(self.data) > 2000: break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item.get('text', '')
        summary = item.get('summary', '')

        # 转换为 token ids
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        summary_ids, _ = tokenizer.encode(summary, maxlen=max_target_len)

        return token_ids, summary_ids

def collate_fn(batch):
    """
    将一个 batch 的数据进行 padding 对齐
    """
    batch_token_ids, batch_summary_ids = [], []
    for token_ids, summary_ids in batch:
        batch_token_ids.append(token_ids)
        batch_summary_ids.append(summary_ids)

    # padding 到同一长度
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_summary_ids = sequence_padding(batch_summary_ids)

    return torch.tensor(batch_token_ids), torch.tensor(batch_summary_ids)

# 加载数据
print("正在加载数据...")
train_dataset = SummaryDataset(train_data_path)
valid_dataset = SummaryDataset(valid_data_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size*2, num_workers=2, pin_memory=True, shuffle=False, collate_fn=collate_fn)

# 构建模型
print("正在加载模型...")

# 读取 Hugging Face 的 config.json
with open(config_path, 'r', encoding='utf-8') as f:
    hf_config = json.load(f)

# 构建参数映射
# 否则会出现 "missing required positional arguments" 错误
bert4torch_args = {
    'model': 'bart', 
    'vocab_size': hf_config['vocab_size'],                  # 词表大小
    'hidden_size': hf_config['d_model'],                    # 对应 d_model
    'num_hidden_layers': hf_config['encoder_layers'],       # 对应 encoder_layers
    'num_attention_heads': hf_config['encoder_attention_heads'], # 对应 encoder_attention_heads
    'intermediate_size': hf_config['encoder_ffn_dim'],      # 对应 encoder_ffn_dim
    'hidden_act': hf_config['activation_function'],         # 对应 activation_function
    'dropout_rate': hf_config['dropout'],
    'max_position': hf_config['max_position_embeddings'],
    'segment_vocab_size': 0,
}

print("已手动完成配置参数映射，开始构建...")

model = build_transformer_model(
    config_path=None, 
    checkpoint_path=checkpoint_path,
    **bert4torch_args
).to(device)

print("模型加载成功！")

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 定义损失函数 (忽略 padding 部分的 loss)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ================= 【新增】定义学习率调度器 =================
# 1. 计算总训练步数 = 每个Epoch的步数 * 总Epoch数
total_steps = len(train_dataloader) * epochs

# 2. 设定预热步数 (Warmup Steps)
warmup_steps = int(total_steps * 0.05)

# 3. 创建调度器
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)
# ==========================================================

# 定义生成器 (用于临时查看效果)
class ArticleSummaryDecoder(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states=None):
        """
        这个函数会被 beam_search 循环调用。
        inputs: 也就是 generate 里传入的 [token_ids]
        output_ids: 目前已经生成的 token 序列
        """
        # 获取编码器输入 (原文)
        token_ids = inputs[0] 
        
        # 调用模型
        logits = model.predict([token_ids, output_ids])
        
        # 获取最后一个 token 的预测结果
        # BART 输出是 [encoder_hidden, decoder_logits]
        # 我们要取列表最后一个(decoder_logits)，然后取序列的最后一步(:, -1, :)
        return logits[-1][:, -1, :]

    def generate(self, text, topk=4):
        """
        生成函数
        topk: 这里表示 Beam Search 的 beam size (集束搜索宽度)，4 是常用值
        """
        # 编码原文
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        
        # 调用 beam_search (它会自动循环调用上面的 predict)
        # 这里的 [token_ids] 就会传给 predict 的 inputs
        output_ids = self.beam_search([token_ids], top_k=topk)

        output_ids = output_ids[0]
        # 解码为文本
        return tokenizer.decode(output_ids)

# 初始化生成器
summary_generator = ArticleSummaryDecoder(
    bos_token_id=tokenizer._token_end_id,
    eos_token_id=tokenizer._token_end_id,
    max_length=max_target_len,
    device=device
)

# 评估
@torch.no_grad()
def evaluate(dataset, limit=100):
    """
    在验证集上跑一遍 ROUGE 评分
    limit: 为了速度，默认只测验证集的前 100 条
    """
    print(f"正在进行 ROUGE 评估 (采样 {limit} 条)...")
    rouge = Rouge()
    preds, refs = [], []
    
    # 切换到评估模式 (关闭 Dropout)
    model.eval()
    
    for i, item in tqdm(enumerate(dataset.data), total=limit, desc="Evaluating"):
        if i >= limit: break
        
        text = item.get('text', '')
        ref = item.get('summary', '')
        
        if not text or not ref: continue
        
        # 生成摘要
        pred = summary_generator.generate(text)
        
        # 中文 ROUGE 需要按字分开
        pred_seg = ' '.join([c for c in pred]) if pred else "空"
        ref_seg = ' '.join([c for c in ref]) if ref else "空"
        
        preds.append(pred_seg)
        refs.append(ref_seg)
    
    # 计算分数
    if preds:
        scores = rouge.get_scores(preds, refs, avg=True)
        return scores
    return None


# 训练
def train():
    print(f"开始训练，设备: {device}")

    start_epoch = 0
    resume_path = os.path.join(project_root, 'model_weights', 'latest_checkpoint.pt')

    if os.path.exists(resume_path):
        print(f"检测到断点文件：{resume_path}，正在加载...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        except Exception as e:
            print(f"加载断点失败，错误信息: {e}")
            print("将从头开始训练。")
    else:
        print("未检测到断点文件，开始新训练。")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for step, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(device).long(), batch_y.to(device).long()
            
            # BART 训练输入:
            # Encoder Input: batch_x (原文)
            # Decoder Input: batch_y[:, :-1] (摘要去掉最后一个字)
            # Label: batch_y[:, 1:] (摘要去掉第一个字，向后移一位)
            
            optimizer.zero_grad()
            
            # bert4torch 的 BART 调用方式：传入 list [src_ids, tgt_ids]
            # 这里我们需要手动构造 decoder input
            decoder_input = batch_y[:, :-1]
            labels = batch_y[:, 1:]
            
            # 前向传播
            model_outputs = model([batch_x, decoder_input])
            logits = model_outputs[-1]
            
            # 计算 Loss (需要把 logits 展平)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # --- 每个 step 更新学习率 ---
            
            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
        
        # --- 每个 Epoch 结束后的工作 ---
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} 训练完成 平均 Loss: {avg_loss:.4f}")
        
        # 保存权重
        weight_save_path = os.path.join(project_root, 'model_weights', f'epoch_{epoch+1}_weights.pt')
        torch.save(model.state_dict(), weight_save_path)
        print(f"模型权重已保存: {weight_save_path}")

        # 保存断点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, resume_path)
        print(f"断点已更新：{resume_path}")

        # 运行评估
        scores = evaluate(valid_dataset)
        
        # 写入json文件
        if scores:
            results_path = os.path.join(project_root, 'results', 'results.jsonl')
            with open(results_path, 'a', encoding='utf-8') as f:
                result_record = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'rouge-1': scores['rouge-1'],
                    'rouge-2': scores['rouge-2'],
                    'rouge-l': scores['rouge-l'],
                }
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            print(f"Epoch {epoch+1} 验证集得分:")
            print(f"   ROUGE-1: {scores['rouge-1']['f'] * 100:.2f}")
            print(f"   ROUGE-2: {scores['rouge-2']['f'] * 100:.2f}")
            print(f"   ROUGE-L: {scores['rouge-l']['f'] * 100:.2f}")


if __name__ == '__main__':
    train()