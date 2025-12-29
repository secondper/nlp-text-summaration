import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 这里填入你控制台打印的数据 =================
epochs = [1, 2, 3, 4, 5]

# 填入每轮的 "平均 Loss"
losses = [4.56, 2.98, 2.39, 1.96, 1.60] 

# 填入每轮验证集的 ROUGE 分数 (如果没有跑完，只填已有的)
rouge_1 = [30.5, 35.2, 38.4, 40.1, 41.5]
rouge_2 = [12.1, 15.8, 18.2, 20.5, 21.8]
rouge_l = [25.4, 30.1, 32.5, 34.8, 36.2]
# ==============================================================

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

# 调整布局防止重叠
plt.tight_layout()

# 保存并显示
plt.savefig('training_analysis.png', dpi=300) # 保存高清图用于报告
print("✅ 图表已生成: training_analysis.png")
plt.show()