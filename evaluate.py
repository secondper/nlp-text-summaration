import json
import os
from rouge import Rouge
from tqdm import tqdm

# ================= 配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 这是你上一步生成的预测文件
prediction_file = os.path.join(current_dir, 'result', 'test_predictions.jsonl')

def load_data(filename):
    """读取预测文件，提取预测结果和标准标签"""
    preds = []
    refs = []
    
    if not os.path.exists(filename):
        print(f"❌ 找不到文件: {filename}")
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

def evaluate():
    # 1. 加载数据
    preds, refs = load_data(prediction_file)
    
    if not preds:
        print("没有读到有效数据，请检查 test_predictions.jsonl 是否生成成功，且包含 'label' 字段。")
        return

    print(f"有效样本数: {len(preds)}")
    print("正在计算 ROUGE 分数...")

    # 2. 初始化 ROUGE 计算器
    rouge = Rouge()
    
    # 3. 计算分数 (avg=True 表示取平均值)
    scores = rouge.get_scores(preds, refs, avg=True)

    # 4. 打印结果
    print("\n" + "="*40)
    print("评估结果 (ROUGE Score)")
    print("="*40)
    
    # 格式化打印
    def print_metric(name, metrics):
        print(f"【{name}】:")
        print(f"  - Recall (召回率):    {metrics['r']*100:.2f}%")
        print(f"  - Precision (准确率): {metrics['p']*100:.2f}%")
        print(f"  - F1-Score (综合分):  {metrics['f']*100:.2f}%")
        print("-" * 20)

    print_metric("ROUGE-1 (单字重合度)", scores['rouge-1'])
    print_metric("ROUGE-2 (词组/二元重合度)", scores['rouge-2'])
    print_metric("ROUGE-L (最长公共子序列/句子结构)", scores['rouge-l'])
    print("="*40)
    
    print("\n💡 结果分析提示：")
    print("ROUGE-1: 衡量信息覆盖度，越高说明关键词抓得越准。")
    print("ROUGE-2: 衡量流畅度，越高说明生成的短语越连贯。")
    print("ROUGE-L: 衡量句子结构相似度。")

if __name__ == "__main__":
    evaluate()