import os
import re
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
    data = funasr_result
    text = data[0]['text']
    timestamps = data[0]['timestamp']

    # 配置参数
    max_chars_per_line = 20  # 每行字幕的最大字符数

    # 首先按照标点符号分割文本为短句
    sentence_pattern = r'([^，。！？,.!?;；、]+[，。！？,.!?;；、]+)'
    phrases = re.findall(sentence_pattern, text)

    # 如果没有找到短句，就把整个文本作为一个短句
    if not phrases:
        phrases = [text]

    # 确保所有文本都被包含
    remaining_text = text
    for phrase in phrases:
        remaining_text = remaining_text.replace(phrase, '', 1)
    if remaining_text.strip():
        phrases.append(remaining_text.strip())

    # 计算每个短句对应的时间戳
    phrase_timestamps = []
    total_chars = len(text)

    char_index = 0
    for phrase in phrases:
        if not phrase.strip():
            continue

        phrase_len = len(phrase)
        # 计算短句在整个文本中的比例
        start_ratio = char_index / total_chars
        end_ratio = (char_index + phrase_len) / total_chars

        start_idx = min(int(start_ratio * len(timestamps)), len(timestamps) - 1)
        end_idx = min(int(end_ratio * len(timestamps)), len(timestamps) - 1)

        if start_idx == end_idx:
            if end_idx < len(timestamps) - 1:
                end_idx += 1

        start_time = timestamps[start_idx][0]
        end_time = timestamps[end_idx][1]

        phrase_timestamps.append((phrase, start_time, end_time))
        char_index += phrase_len

    # 合并短句为合适长度的字幕段落，只考虑字数限制
    text_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for phrase, start, end in phrase_timestamps:
        # 如果当前段落为空，直接添加
        if not current_text:
            current_text = phrase
            current_start = start
            current_end = end
            continue

        # 检查添加当前短句后是否会超出字数限制
        combined_text = current_text + phrase

        if len(combined_text) > max_chars_per_line:
            # 如果会超出限制，保存当前段落并开始新段落
            text_segments.append((current_text, current_start, current_end))
            current_text = phrase
            current_start = start
            current_end = end
        else:
            # 否则合并短句
            current_text = combined_text
            current_end = end

    # 添加最后一个段落
    if current_text:
        text_segments.append((current_text, current_start, current_end))

    # 生成SRT格式，去除每段末尾的标点符号
    srt = ""
    for i, (text, start, end) in enumerate(text_segments, 1):
        # 去除段落末尾的标点符号
        cleaned_text = re.sub(r'[，。！？,.!?;；、]+$', '', text)

        srt += f"{i}\n"
        srt += f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
        srt += f"{cleaned_text.strip()}\n\n"

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
            srt = funasr_to_srt(result)
            result[0]['srt'] = srt
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
