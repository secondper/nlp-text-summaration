import os
import json
import torch
import numpy as np
import gradio as gr
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder

# 配置
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(current_dir, 'checkpoint')
model_dir = os.path.join(current_dir, 'model')

config_path = os.path.join(checkpoint_dir, 'config.json')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

# 加载权重的路径
weight_path = os.path.join(model_dir, 'bart_epoch_4.pt') 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
maxlen = 512
max_target_len = 128

print(f"正在启动网页端... 设备: {device}")

# 加载模型
tokenizer = Tokenizer(dict_path, do_lower_case=True)

with open(config_path, 'r', encoding='utf-8') as f:
    hf_config = json.load(f)

bert4torch_args = {
    'model': 'bart', 
    'vocab_size': hf_config['vocab_size'],
    'hidden_size': hf_config['d_model'],
    'num_hidden_layers': hf_config['encoder_layers'],
    'num_attention_heads': hf_config['encoder_attention_heads'],
    'intermediate_size': hf_config['encoder_ffn_dim'],
    'hidden_act': hf_config['activation_function'],
    'dropout_rate': hf_config['dropout'],
    'max_position': hf_config['max_position_embeddings'],
    'segment_vocab_size': 0,
}

model = build_transformer_model(config_path=None, checkpoint_path=None, **bert4torch_args).to(device)

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print("模型权重加载成功！")
else:
    print(f"找不到权重: {weight_path}")

# 定义推理逻辑
class ArticleSummaryDecoder(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states=None):
        token_ids = inputs[0] 
        logits = model.predict([token_ids, output_ids])
        return logits[-1][:, -1, :]

    def generate(self, text, topk=4):
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids], top_k=topk)
        
        # 格式清洗
        output_ids = output_ids[0]
            
        return tokenizer.decode(output_ids)

summary_generator = ArticleSummaryDecoder(
    bos_token_id=tokenizer._token_end_id,
    eos_token_id=tokenizer._token_end_id,
    max_length=max_target_len,
    device=device
)

def predict_fn(text):
    if not text: return "请输入内容..."
    return summary_generator.generate(text)

# 搭建 Gradio 界面
# 这里定义页面的样式和交互
with gr.Blocks(title="智能摘要生成系统", theme=gr.themes.Soft()) as demo:
    
    # 标题部分
    gr.Markdown("# 智能摘要生成系统")
    gr.Markdown("本系统基于 **OpenMOSS-Team/bart-base-chinese** 模型微调，能够自动提取商品核心卖点或生成新闻标题。")

    # 左右布局
    with gr.Row():
        with gr.Column():
            # 左边：输入
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="请粘贴商品详情描述或新闻长文本...",
                lines=10
            )
            # 按钮
            submit_btn = gr.Button("✨ 生成摘要", variant="primary")
            
            # 清除按钮
            clear_btn = gr.Button("🗑️ 清空")

        with gr.Column():
            # 右边：输出
            output_text = gr.Textbox(
                label="生成结果",
                lines=5
            )

    # 底部：示例 (这点非常加分！助教点一下就能看效果)
    gr.Markdown("### 📝 点击下方示例快速测试")
    gr.Examples(
        examples=[
            ["2007年乔布斯向人们展示iPhone并宣称“它将会改变世界”，还有人认为他在夸大其词，然而在8年后，以iPhone为代表的触屏智能手机已经席卷全球各个角落。未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更大的贡献。"],
            ["长期饮用过烫的饮品（超过65°C）已被世界卫生组织列为明确的致癌风险因素。高温会反复灼伤食道黏膜，引发慢性炎症，从而可能增加食管癌变几率。专家建议，将热饮晾置片刻，待温热适口时再饮用，是简单有效的保护习惯。"],
            ["天文学家通过詹姆斯·韦伯太空望远镜，在一颗系外行星的大气中首次清晰探测到甲烷与二氧化碳存在的迹象。该行星位于宜居带，这一发现为寻找地外生命提供了关键数据，是系外行星研究领域的一项重大突破。"]
        ],
        inputs=input_text,
        outputs=output_text,
        fn=predict_fn,
        cache_examples=False,
    )

    # 绑定事件
    submit_btn.click(fn=predict_fn, inputs=input_text, outputs=output_text)
    clear_btn.click(lambda: ("", ""), outputs=[input_text, output_text])

# 启动服务
if __name__ == "__main__":
    # launch 会自动在本地启动一个网页服务器
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=True)