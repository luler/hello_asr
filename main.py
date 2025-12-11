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



PUNCS = set("，。！？,.!?;；、")


def build_char2ts_index(text: str, timestamps):
    """
    根据文本字符顺序（忽略标点和空白）为每个可发音字符分配一个时间戳索引。
    假设 timestamps 按 token/字顺序给出。
    """
    char_positions = [
        i for i, ch in enumerate(text)
        if ch not in PUNCS and not ch.isspace()
    ]

    char2ts = {}
    n = min(len(char_positions), len(timestamps))
    for ts_idx in range(n):
        char_idx = char_positions[ts_idx]
        char2ts[char_idx] = ts_idx

    return char2ts


def split_phrases_with_pos(text: str):
    """
    按标点切句，并保留每句在原文中的字符起止位置。
    返回: [(phrase_text, char_start, char_end), ...]
    """
    phrases = []
    start = 0
    for i, ch in enumerate(text):
        if ch in PUNCS:
            if i >= start:
                phrases.append((text[start:i + 1], start, i))
            start = i + 1

    if start < len(text):
        phrases.append((text[start:], start, len(text) - 1))

    return phrases


def phrases_to_time_segments(text: str, timestamps):
    """
    核心：按“句子”映射到一段连续的 token 时间，
    并保证同一个 token 不会被前后两句同时使用。
    """
    char2ts = build_char2ts_index(text, timestamps)
    phrases = split_phrases_with_pos(text)

    segments = []
    last_ts_end = -1  # 上一条字幕用到的最大 token index

    for phrase, c_start, c_end in phrases:
        # 只使用 ts_index > last_ts_end 的 token，避免复用
        ts_indices = [
            char2ts[i]
            for i in range(c_start, c_end + 1)
            if i in char2ts and char2ts[i] > last_ts_end
        ]
        if not ts_indices:
            # 这一句可能全是标点，或者对应 token 已被前面吃完
            continue

        ts_start = min(ts_indices)
        ts_end = max(ts_indices)

        start_time = timestamps[ts_start][0]
        end_time = timestamps[ts_end][1]

        segments.append((phrase, start_time, end_time))
        last_ts_end = ts_end

    return segments


def funasr_to_srt(funasr_result):
    """
    将 funasr 的识别结果转换为 SRT 字幕：
    - 使用词级时间戳，线性扫描，不复用 token；
    - 再按字数合并成合适长度的字幕段。
    """
    data = funasr_result
    if not data:
        return ""

    text = data[0]['text']
    timestamps = data[0]['timestamp']

    # 1. 按句子拿到基础时间段（token 顺序 + 不复用）
    phrase_segments = phrases_to_time_segments(text, timestamps)
    # phrase_segments: [(phrase_text, start_ms, end_ms), ...]

    # 2. 按字数合并成字幕段
    max_chars_per_line = 20  # 每条字幕的最大字符数，可按需调整

    text_segments = []
    current_text = ""
    current_start = None
    current_end = None

    for phrase, start_ms, end_ms in phrase_segments:
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            continue

        if not current_text:
            # 当前字幕为空，直接起一条新字幕
            current_text = cleaned_phrase
            current_start = start_ms
            current_end = end_ms
            continue

        # 尝试把当前短句拼接到这一条字幕里
        combined_text = current_text + cleaned_phrase
        if len(combined_text) > max_chars_per_line:
            # 超过字数限制，先收掉当前字幕，再起新字幕
            text_segments.append((current_text, current_start, current_end))
            current_text = cleaned_phrase
            current_start = start_ms
            current_end = end_ms
        else:
            # 不超，就合并到同一条字幕
            current_text = combined_text
            current_end = max(current_end, end_ms)

    # 收尾：把最后一条字幕加进去
    if current_text:
        text_segments.append((current_text, current_start, current_end))

    # 3. 时间轴校正：保证严格单调、无重叠
    MIN_GAP = 0     # 相邻字幕的最小间隔（毫秒），不需要就设 0
    MIN_DUR = 300   # 单条字幕最短显示时间（毫秒）

    fixed_segments = []
    prev_end = 0
    for seg_text, start, end in text_segments:
        if start < prev_end + MIN_GAP:
            start = prev_end + MIN_GAP
        if end <= start:
            end = start + MIN_DUR
        fixed_segments.append((seg_text, start, end))
        prev_end = end

    # 4. 生成 SRT 字符串
    srt_lines = []
    for i, (seg_text, start, end) in enumerate(fixed_segments, 1):
        # 去掉末尾多余标点
        cleaned_text = re.sub(r'[，。！？,.!?;；、]+$', '', seg_text).strip()
        if not cleaned_text:
            continue

        srt_lines.append(str(i))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(cleaned_text)
        srt_lines.append("")  # 空行分隔

    return "\n".join(srt_lines)


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
