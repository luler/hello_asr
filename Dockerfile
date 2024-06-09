# 使用官方的 Python 镜像作为基础镜像
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt requirements.txt

# 安装 Python 依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录中的文件到工作目录中
COPY . .

# 下载模型
RUN python -c "from funasr import AutoModel; AutoModel(model='paraformer-zh',vad_model='fsmn-vad',punc_model='ct-punc')"

# 暴露端口
EXPOSE 12369

# 启动 FastAPI 应用程序
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "12369"]