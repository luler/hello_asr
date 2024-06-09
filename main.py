import os
import tempfile
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from funasr import AutoModel

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device=device,
    # spk_model="cam++",
)


# 异步函数，用于保存上传的文件到临时目录
async def save_upload_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[1]  # 获取文件后缀名
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:  # 创建临时文件
        temp_file.write(await upload_file.read())  # 将上传的文件内容写入临时文件
        return temp_file.name  # 返回临时文件路径


@app.post("/asr")
async def asr(file: List[UploadFile] = File(...)):
    temp_input_file_path = None
    try:
        if not file or any(f.filename == "" for f in file):
            raise Exception("No file was uploaded")
        if len(file) != 1:
            raise Exception("Only one file can be uploaded at a time")
        file = file[0]

        ext_name = os.path.splitext(file.filename)[1].strip('.')
        if ext_name not in ['wav', 'mp3']:
            raise Exception("Unsupported file extension")
        temp_input_file_path = await save_upload_file(file)  # 保存上传的文件

        result = model.generate(
            input=temp_input_file_path,
            batch_size_s=300,
            # hotword='魔搭'
        )

        return {"result": result}  # 返回识别结果
    except Exception as e:  # 捕获其他异常
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        for temp_file in [temp_input_file_path]:
            if temp_file and os.path.exists(temp_file):  # 检查路径是否存在
                os.remove(temp_file)  # 删除文件


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12369)  # 运行FastAPI应用
