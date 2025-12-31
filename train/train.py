import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
from rouge import Rouge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.generation import AutoRegressiveDecoder

# path setting
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.model import get_bart_model
from core.dataset import SummaryDataset, collate_fn
from core.decoder import ArticleSummaryDecoder
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="BART model finetune for text summarization")
    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help="warmup steps ratio")
    # model parameters
    parser.add_argument('--maxlen', type=int, default=512, help="max length of source text")
    parser.add_argument('--max_target_len', type=int, default=128, help="max length of summary")
    parser.add_argument('--train_datasize', type=int, default=20000, help="size of training data to use")
    parser.add_argument('--valid_datasize', type=int, default=2000, help="size of validation data to use")
    # path configuration
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data', 'LCSTS_origin'), help="training data path")
    parser.add_argument("--save_dir", type=str, default=os.path.join(project_root, 'model_weights'), help="model weights save path")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(project_root, 'checkpoint'), help="pretrained model folder path")
    # others
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="training device")
    parser.add_argument("--resume", action="store_true", help="whether to resume training from checkpoint")

    return parser.parse_args()

# training function
def train(args):
    # print current configuration
    print("-" * 30)
    print("Experiment Configuration:\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 30)
    # path
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    checkpoint_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
    dict_path = os.path.join(args.checkpoint_dir, 'vocab.txt')

    train_data_path = os.path.join(args.data_dir, 'train.jsonl')
    valid_data_path = os.path.join(args.data_dir, 'valid.jsonl')
    # ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # set random seed
    seed_everything(args.seed)
    # tokenizer
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # loading data
    print("Loading data...")
    train_dataset = SummaryDataset(tokenizer, train_data_path, maxlen=args.maxlen, max_target_len=args.max_target_len, train_datasize=args.train_datasize)
    valid_dataset = SummaryDataset(tokenizer, valid_data_path, maxlen=args.maxlen, max_target_len=args.max_target_len, valid_datasize=args.valid_datasize)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn)

    # build model
    print("Loading model...")  
    model = get_bart_model(config_path=config_path, checkpoint_path=checkpoint_path, device=args.device)
    print("Model loaded successfully!")

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # define loss function (ignore padding loss)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # define learning rate scheduler
    # total training steps = steps per epoch * total epochs
    total_steps = len(train_dataloader) * args.epochs
    # set warmup steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    # create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # initialize generator
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
        print(f"Detected checkpoint file: {resume_path}, loading...")
        try:
            checkpoint = torch.load(resume_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        except Exception as e:
            print(f"Failed to load checkpoint, error: {e}\n Restarting from scratch")
    
    print(f"Starting training on device: {args.device}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for step, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x, batch_y = batch_x.to(args.device).long(), batch_y.to(args.device).long()
            
            # BART training inputs:
            # Encoder Input: batch_x (original text)
            # Decoder Input: batch_y[:, :-1] (summary without the last token)
            # Label: batch_y[:, 1:] (summary without the first token, shifted right)
            
            optimizer.zero_grad()
            
            # bert4torch's BART call method: pass in list [src_ids, tgt_ids]
            # Here we need to manually construct decoder input
            decoder_input = batch_y[:, :-1]
            labels = batch_y[:, 1:]
            
            # forward pass
            model_outputs = model([batch_x, decoder_input])
            logits = model_outputs[-1]
            
            # calculate loss (need to flatten logits)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Work to do at the end of each Epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} training completed. Average Loss: {avg_loss:.4f}")
        
        # save weights
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'bart_epoch_{epoch+1}.pt'))

        # save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, resume_path)
        print(f"Checkpoint updated: {resume_path}")

        # evaluate on valid_dataset
        scores = evaluate(model, summary_generator, valid_dataset)
        
        # write to json file
        if scores:
            results_path = os.path.join(project_root, 'results', 'results.jsonl')
             # Ensure the results directory exists
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
            print(f"Epoch {epoch+1} validation scores:")
            print(f"   ROUGE-1: {scores['rouge-1']['f'] * 100:.2f}")
            print(f"   ROUGE-2: {scores['rouge-2']['f'] * 100:.2f}")
            print(f"   ROUGE-L: {scores['rouge-l']['f'] * 100:.2f}")

if __name__ == '__main__':
    args = parse_args()
    train(args)
