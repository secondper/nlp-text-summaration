import os
import sys
import torch
import gradio as gr
from bert4torch.tokenizers import Tokenizer

# 配置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.model import get_bart_model
from core.decoder import ArticleSummaryDecoder


checkpoint_dir = os.path.join(current_dir, 'checkpoint')
config_path = os.path.join(checkpoint_dir, 'config.json')
dict_path = os.path.join(checkpoint_dir, 'vocab.txt')

# 定义权重路径
weights_path = os.path.join(current_dir, 'model_weights', 'bart_epoch_10.pt')

# 超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
maxlen = 512        # 原文最大长度
max_target_len = 128 # 生成摘要最大长度

print(f"正在启动网页端... 设备: {device}")

def init_system():
    """初始化分词器、模型和生成器"""
    print("正在加载分词器...")
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    print("正在构建模型结构...")
    model = get_bart_model(config_path=config_path, checkpoint_path=None, device=device)
    print("模型构建成功！")

    print(f"正在加载微调权重: {weights_path}")
    if os.path.exists(weights_path):
        # 加载权重
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval() # 切换到评估模式
        print("模型权重加载成功！")
    else:
        print(f"警告: 找不到权重文件 {weights_path}，将使用随机初始化模型（输出将是乱码）。")

    # 初始化生成器 (使用 core.decoder 中的类)
    generator = ArticleSummaryDecoder(
        model=model,
        tokenizer=tokenizer,
        bos_token_id=tokenizer._token_end_id,
        eos_token_id=tokenizer._token_end_id,
        max_length=max_target_len,
        device=device
    )
    
    return generator

# 全局初始化和生成器
summary_generator = init_system()

def predict_fn(text):
    """Gradio 调用的核心预测函数"""
    if not text or not text.strip():
        return "请输入有效的新闻文本..."
    
    try:
        # 调用生成器的 generate 方法
        summary = summary_generator.generate(text, maxlen=maxlen, topk=4)
        return summary
    except Exception as e:
        return f"生成出错: {str(e)}"

def build_interface():
    with gr.Blocks(title="新闻摘要智能生成系统") as demo:
        
        # --- 头部 ---
        gr.Markdown("# 📰 新闻摘要智能生成系统")
        gr.Markdown("""
        本系统基于 **BART (Bidirectional and Auto-Regressive Transformers)** 架构，
        使用 🤗 [OpenMOSS-Team/bart-base-chinese](https://huggingface.co/OpenMOSS-Team/bart-base-chinese) 进行微调开发。
        """)

        # --- 主体区域 (左右分栏) ---
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="输入文本",
                    placeholder="请粘贴新闻文本...",
                    lines=12,
                    max_lines=20
                )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空内容", variant="secondary")
                    submit_btn = gr.Button("✨ 生成摘要", variant="primary")

            # 右侧：输出区
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="生成摘要",
                    placeholder="AI 生成的结果将显示在这里...",
                    lines=6,
                    buttons=["copy"], # 允许一键复制
                    interactive=False      # 输出框不可编辑
                )

        # --- 底部：示例区 ---
        gr.Markdown("### ⚡ 点击示例快速体验")
        gr.Examples(
            examples=[
                ["2007年乔布斯向人们展示iPhone并宣称“它将会改变世界”，还有人认为他在夸大其词，然而在8年后，以iPhone为代表的触屏智能手机已经席卷全球各个角落。未来，智能手机将会成为“真正的个人电脑”，为人类发展做出更大的贡献。"],
                ["长期饮用过烫的饮品（超过65°C）已被世界卫生组织列为明确的致癌风险因素。高温会反复灼伤食道黏膜，引发慢性炎症，从而可能增加食管癌变几率。专家建议，将热饮晾置片刻，待温热适口时再饮用，是简单有效的保护习惯。"],
                ["天文学家通过詹姆斯·韦伯太空望远镜，在一颗系外行星的大气中首次清晰探测到甲烷与二氧化碳存在的迹象。该行星位于宜居带，这一发现为寻找地外生命提供了关键数据，是系外行星研究领域的一项重大突破。"],
                ["著名中国文学评论家夏志清29日在美去世。有评论认为，夏先生1961年出版的英语著作《中国现代小说史》是中国现代小说批评的拓荒巨著，不亚于一次文学革命。此书之后，中国现代文学研究才进入西方高校。在夏志清看来，《金锁记》是中国文学史上最好的小说。"],
                ["步入深水区的房地产调控政策走向，再度引发官媒聚焦。15日，新华社旗下《经济参考报》报道称，相关内部会议透露，将加快研究包括土地、金融、财税等方面的房地产中长期调控政策。“去行政化”将成为未来调控方向。"],
            ],
            inputs=input_text,
            outputs=output_text,
            fn=predict_fn,
            cache_examples=False, # 设为 False 加快启动速度
        )

        # --- 事件绑定 ---
        submit_btn.click(
            fn=predict_fn, 
            inputs=input_text, 
            outputs=output_text
        )
        
        clear_btn.click(
            fn=lambda: ("", ""), # 清空输入和输出
            inputs=None, 
            outputs=[input_text, output_text]
        )

    return demo

# ================= 5. 启动入口 =================
if __name__ == "__main__":
    demo = build_interface()
    # allowed_paths 允许访问本地文件（如果需要展示图片等）
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        inbrowser=True,
        theme=gr.themes.Soft(),
        share=False # 如果需要生成公网链接分享给别人，改为 True
    )