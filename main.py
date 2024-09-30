import os
import tempfile
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from funasr import AutoModel
import json

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="ct-punc",
    device=device,
    # spk_model="cam++",
)


def convert_audio(input_file):
    import ffmpeg

    output_file = input_file + ".wav"
    (
        ffmpeg.input(input_file)
        .output(output_file)
        .run(quiet=True)
    )
    return output_file

# 异步函数，用于保存上传的文件到临时目录
async def save_upload_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[1]  # 获取文件后缀名
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:  # 创建临时文件
        temp_file.write(await upload_file.read())  # 将上传的文件内容写入临时文件
        return temp_file.name  # 返回临时文件路径


def funasr_to_srt(funasr_result):
    # 解析JSON字符串
    # data = json.loads(funasr_result)
    data=funasr_result
    subtitles = data[0]['text'].split('. ')
    timestamps = data[0]['timestamp']
    
    # 初始化SRT字符串
    srt = ""
    subtitle_index = 1
    
    # 遍历字幕和时间戳
    for i in range(len(subtitles)):
        if subtitles[i]:  # 确保字幕不为空
            # 格式化时间戳
            start_time = timestamps[i][0]
            end_time = timestamps[i][1]
            srt += f"{subtitle_index}\n"
            srt += f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
            srt += f"{subtitles[i]}.\n\n"
            subtitle_index += 1
    
    return srt

def format_timestamp(milliseconds):
    # 将毫秒转换为SRT格式的时间戳
    seconds, milliseconds = divmod(milliseconds, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


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

        temp_input_file_path = await save_upload_file(file)  # 保存上传的文件
        if ext_name not in ['wav', 'mp3']:
            # 如果不是音频文件,用ffmpeg转换为音频文件
            temp_input_file_path = convert_audio(temp_input_file_path)
            # raise Exception("Unsupported file extension")

        print(temp_input_file_path)

        result = model.generate(
            input=temp_input_file_path,
            batch_size_s=300,
            batch_size_threshold_s=60,
            # hotword='魔搭'
        )

        try:
            srt=funasr_to_srt(result)
            result[0]['srt']=srt
        except:
            print('srt convert fail')

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
