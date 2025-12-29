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
import sys
import argparse
import numpy as np
from rouge import Rouge
from transformers import get_linear_schedule_with_warmup

# 配置参数
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.model import get_bart_model
from core.dataset import SummaryDataset, collate_fn
from core.decoder import ArticleSummaryDecoder
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="BART 模型微调训练脚本")
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help="预热步数比例")
    # 模型参数
    parser.add_argument('--maxlen', type=int, default=512, help="原文最大长度")
    parser.add_argument('--max_target_len', type=int, default=128, help="摘要最大长度")
    # 路径配置
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data', 'LCSTS_origin'), help="训练数据路径")
    parser.add_argument("--save_dir", type=str, default=os.path.join(project_root, 'model_weights'), help="模型权重保存路径")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(project_root, 'checkpoint'), help="预训练模型文件夹路径")
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--resume", action="store_true", help="是否从断点继续训练")

    return parser.parse_args()

# 训练
def train(args):
    # 打印当前配置
    print("-" * 30)
    print("实验配置：\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 30)
    # 路径
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    checkpoint_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
    dict_path = os.path.join(args.checkpoint_dir, 'vocab.txt')

    train_data_path = os.path.join(args.data_dir, 'train.jsonl')
    valid_data_path = os.path.join(args.data_dir, 'valid.jsonl')
    # 确保保存目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 设置随机种子
    seed_everything(args.seed)
    # 分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # 加载数据
    print("正在加载数据...")
    train_dataset = SummaryDataset(tokenizer, train_data_path, maxlen=args.maxlen, max_target_len=args.max_target_len, train_datasize=20, valid_datasize=2)
    valid_dataset = SummaryDataset(tokenizer, valid_data_path, maxlen=args.maxlen, max_target_len=args.max_target_len, train_datasize=20, valid_datasize=2)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn)

    # 构建模型
    print("正在加载模型...")  
    model = get_bart_model(config_path=config_path, checkpoint_path=checkpoint_path, device=args.device)
    print("模型加载成功！")

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 定义损失函数 (忽略 padding 部分的 loss)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 定义学习率调度器
    # 总训练步数 = 每个Epoch的步数 * 总Epoch数
    total_steps = len(train_dataloader) * args.epochs
    # 设定预热步数 (Warmup Steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    # 创建调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # 初始化生成器
    summary_generator = ArticleSummaryDecoder(
        model=model,
        tokenizer=tokenizer,
        bos_token_id=tokenizer._token_end_id,
        eos_token_id=tokenizer._token_end_id,
        max_length=args.max_target_len,
        device=args.device
    )

    start_epoch = 0
    resume_path = os.path.join(args.save_dir, 'latest_checkpoint.pt')
    if args.resume and os.path.exists(resume_path):
        print(f"检测到断点文件：{resume_path}，正在加载...")
        try:
            checkpoint = torch.load(resume_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        except Exception as e:
            print(f"加载断点失败，错误信息: {e}\n 重新开始")
    
    print(f"开始训练，设备: {args.device}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(args.device).long(), batch_y.to(args.device).long()
            
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
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
        
        # --- 每个 Epoch 结束后的工作 ---
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} 训练完成 平均 Loss: {avg_loss:.4f}")
        
        # 保存权重
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'bart_epoch_{epoch+1}.pt'))

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
        scores = evaluate(model, summary_generator, valid_dataset)
        
        # 写入json文件
        if scores:
            results_path = os.path.join(project_root, 'results', 'results.jsonl')
             # 确保 results 目录存在
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
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
    args = parse_args()
    train(args)
