import json
import os

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_dir, 'data', 'LCSTS_origin', 'train.jsonl')
my_test_file = os.path.join(current_dir, 'data', 'LCSTS_origin', 'my_test.jsonl')

print("正在切分数据...")
lines = []
with open(train_file, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    test_lines = all_lines[-200:] # 取最后200条

with open(my_test_file, 'w', encoding='utf-8') as f:
    for line in test_lines:
        f.write(line)

print(f"已生成自定义测试集: {my_test_file}")