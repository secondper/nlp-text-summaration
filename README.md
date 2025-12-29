pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

#   基于BART的摘要生成

##  项目介绍
本项目属于USTC-2025秋深度学习实践课程大作业

##  环境依赖

请配置以下环境
- python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- bert4torch

安装命令
```
pip freeze > requirements.txt
pip install -r requirements.txt
```


修改源码
build_transformer_model转到返回类型中的transformer转到第237行，添加代码
```
# 修改后：如果 decoder_input 不是 list，就把它变成 list
        if not isinstance(decoder_input, list):
            decoder_input = [decoder_input]
```
可以跑通

下面考虑加补丁
做不到


