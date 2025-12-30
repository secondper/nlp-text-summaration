import os

# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
train_file = os.path.join(project_root, 'data', 'LCSTS_origin', 'train.jsonl')
my_test_file = os.path.join(project_root, 'data', 'LCSTS_origin', 'my_test.jsonl')

print("spliting data...")
lines = []
with open(train_file, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    test_lines = all_lines[-200:] # take the last 200 lines
with open(my_test_file, 'w', encoding='utf-8') as f:
    for line in test_lines:
        f.write(line)

print(f"Custom test dataset created: {my_test_file}")