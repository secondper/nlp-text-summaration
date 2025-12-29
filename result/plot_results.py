import matplotlib.pyplot as plt
import json
import os

# 指定结果文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file_path = os.path.join(current_dir, 'results.jsonl')

# 读取训练结果数据
def read_results(file_path):
    epochs = []
    losses = []
    rouge_1 = []
    rouge_2 = []
    rouge_l = []

    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return epochs, losses, rouge_1, rouge_2, rouge_l
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    epochs.append(record['epoch'])
                    losses.append(record['loss'])
                    rouge_1.append(record['rouge-1'])
                    rouge_2.append(record['rouge-2'])
                    rouge_l.append(record['rouge-l'])
        print(f"成功读取文件：{file_path}")
        return epochs, losses, rouge_1, rouge_2, rouge_l
    except Exception as e:
        print(f"读取文件时出错：{e}")

epochs, losses, rouge_1, rouge_2, rouge_l = read_results(result_file_path)

# 设置中文字体 (防止中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 自带黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 创建画布 (宽12，高5)
plt.figure(figsize=(12, 5))

# --- 图1: 训练 Loss 曲线 ---
plt.subplot(1, 2, 1) # 1行2列的第1张图
plt.plot(epochs, losses, 'o-', color='#FF5733', label='Training Loss', linewidth=2)
plt.title('模型训练 Loss 变化曲线', fontsize=14)
plt.xlabel('Epoch (训练轮次)', fontsize=12)
plt.ylabel('Loss (损失值)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- 图2: 验证集 ROUGE 分数曲线 ---
plt.subplot(1, 2, 2) # 1行2列的第2张图
plt.plot(epochs, rouge_1, 's-', label='ROUGE-1', color='#1f77b4')
plt.plot(epochs, rouge_l, '^-', label='ROUGE-L', color='#2ca02c')
plt.plot(epochs, rouge_2, 'd-', label='ROUGE-2', color='#ff7f0e')

plt.title('验证集 ROUGE 指标评估', fontsize=14)
plt.xlabel('Epoch (训练轮次)', fontsize=12)
plt.ylabel('Score (F1-Score)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 设置Y轴范围，让图表更好看
plt.ylim([min(min(rouge_1), min(rouge_2), min(rouge_l)) * 0.9, 
          max(max(rouge_1), max(rouge_2), max(rouge_l)) * 1.1])

# 调整布局防止重叠
plt.tight_layout()

# 保存并显示
plt.savefig('training_analysis.png', dpi=300) # 保存高清图用于报告
print("✅ 图表已生成: training_analysis.png")