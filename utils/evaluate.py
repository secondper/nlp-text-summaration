import torch
from rouge import Rouge
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, summary_generator, dataset, limit=100):
    print(f"正在进行 ROUGE 评估 (采样 {limit} 条)...")
    rouge = Rouge()
    preds, refs = [], []
    model.eval()
    
    for i, item in tqdm(enumerate(dataset.data), total=limit, desc="Evaluating"):
        if i >= limit: break
        text = item.get('text', '')
        ref = item.get('summary', '')
        if not text or not ref: continue
        
        pred = summary_generator.generate(text)
        pred_seg = ' '.join([c for c in pred]) if pred else "空"
        ref_seg = ' '.join([c for c in ref]) if ref else "空"
        
        preds.append(pred_seg)
        refs.append(ref_seg)
    
    if preds:
        return rouge.get_scores(preds, refs, avg=True)
    return None