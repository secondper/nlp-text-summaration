import os
import bert4torch

def fix_bert4torch_transformer():
    """
    自动定位并修复 bert4torch 库中 transformer.py 的兼容性 Bug
    """
    lib_path = os.path.dirname(bert4torch.__file__)
    target_file = os.path.join(lib_path, 'models', 'transformer.py')
    
    print(f"正在定位文件: {target_file}")
    
    if not os.path.exists(target_file):
        print("错误：找不到 transformer.py 文件！")
        return

    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找并替换 Bug 代码行
    # 源代码：decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])
    # 改成：decoder_outputs = self.decoder((decoder_input if isinstance(decoder_input, list) else [decoder_input]) + [encoder_hidden_states, encoder_attention_mask])
    
    bug_signature = "decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])"
    fixed_code = "decoder_outputs = self.decoder((decoder_input if isinstance(decoder_input, list) else [decoder_input]) + [encoder_hidden_states, encoder_attention_mask])"
    
    found = False
    for i, line in enumerate(lines):
        if bug_signature in line.strip():
            print(f"发现 Bug 代码行 (Line {i+1}): {line.strip()}")

            indent = line[:line.find(bug_signature.strip())] # 获取原有的缩进
            lines[i] = indent + fixed_code.strip() + "\n"
            found = True
            break
   
    if found:
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("修复成功！bert4torch 库文件已更新")
            print("现在可以正常运行 train.py 了")
        except PermissionError:
            print("权限不足！请尝试以管理员身份运行此脚本。")
    else:
        print("未找到目标代码行，可能你的 bert4torch 版本较新或已被修改。")

if __name__ == "__main__":
    fix_bert4torch_transformer()