import json
import os
from rouge import Rouge
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
# 预测的文件
prediction_file = os.path.join(project_root, 'results', 'test_predictions.jsonl')

def load_data(filename):
    """读取预测文件，提取预测结果和标准标签"""
    preds = []
    refs = []
    
    if not os.path.exists(filename):
        print(f"找不到文件: {filename}")
        return [], []

    print(f"正在读取文件: {filename} ...")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
                p = item.get('predict', '')
                r = item.get('label', '') # 之前的脚本里我们把参考摘要存为了 label

                # 过滤掉空的数据，防止报错
                if p and r:
                    # 【关键步骤】中文评测需要把每个字用空格隔开
                    # 否则 rouge 会把整句话当成一个单词
                    p_seg = ' '.join([char for char in p])
                    r_seg = ' '.join([char for char in r])
                    
                    preds.append(p_seg)
                    refs.append(r_seg)
            except:
                continue
    
    return preds, refs

def scores_compute():
    # 加载数据
    preds, refs = load_data(prediction_file)
    
    if not preds:
        print("没有读到有效数据，请检查 test_predictions.jsonl 是否生成成功，且包含 'label' 字段。")
        return

    print(f"有效样本数: {len(preds)}")
    print("正在计算 ROUGE 分数...")

    # 初始化 ROUGE 计算器
    rouge = Rouge()
    
    # 计算分数
    scores = rouge.get_scores(preds, refs, avg=True)

    # 写入txt文件
    output_txt = os.path.join(project_root, 'results', 'rouge_scores.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("评估结果 (ROUGE Score)\n")
        f.write("="*40 + "\n")
        
        def write_metric(name, metrics):
            f.write(f"【{name}】:\n")
            f.write(f"  - Recall (召回率):    {metrics['r']*100:.2f}%\n")
            f.write(f"  - Precision (准确率): {metrics['p']*100:.2f}%\n")
            f.write(f"  - F1-Score (综合分):  {metrics['f']*100:.2f}%\n")
            f.write("-" * 20 + "\n")

        write_metric("ROUGE-1 (单字重合度)", scores['rouge-1'])
        write_metric("ROUGE-2 (词组/二元重合度)", scores['rouge-2'])
        write_metric("ROUGE-L (最长公共子序列/句子结构)", scores['rouge-l'])
        f.write("="*40 + "\n")
        
        f.write("\n结果分析提示：\n")
        f.write("ROUGE-1: 衡量信息覆盖度，越高说明关键词抓得越准。\n")
        f.write("ROUGE-2: 衡量流畅度，越高说明生成的短语越连贯。\n")
        f.write("ROUGE-L: 衡量句子结构相似度。\n")
    print(f"ROUGE 分数已保存至: {output_txt}")

if __name__ == "__main__":
    scores_compute()