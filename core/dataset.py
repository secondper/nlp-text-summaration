import json
import torch
from torch.utils.data import Dataset
from bert4torch.snippets import sequence_padding

class SummaryDataset(Dataset):
    """
    text summarization dataset
    Read data file in jsonl format, each line is a json object containing 'text' and 'summary' fields.
    Encode text using tokenizer and return token ids.
    """
    def __init__(self, tokenizer, file_path, maxlen=512, max_target_len=128, train_datasize=20000, valid_datasize=2000):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.max_target_len = max_target_len
        self.train_datasize = train_datasize
        self.valid_datasize = valid_datasize
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                if "train.jsonl" in file_path and len(self.data) > self.train_datasize: break 
                if "valid.jsonl" in file_path and len(self.data) > self.valid_datasize: break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item.get('text', '')
        summary = item.get('summary', '')

        token_ids, _ = self.tokenizer.encode(text, maxlen=self.maxlen)
        summary_ids, _ = self.tokenizer.encode(summary, maxlen=self.max_target_len)
        return token_ids, summary_ids

def collate_fn(batch):
    batch_token_ids, batch_summary_ids = [], []
    for token_ids, summary_ids in batch:
        batch_token_ids.append(token_ids)
        batch_summary_ids.append(summary_ids)

    batch_token_ids = sequence_padding(batch_token_ids)
    batch_summary_ids = sequence_padding(batch_summary_ids)
    return torch.tensor(batch_token_ids), torch.tensor(batch_summary_ids)
