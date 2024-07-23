import httpx
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def ask_llm(srt: str, tgtlang='中文', model='gpt-4o', timeout=5*60):
    api_url = os.environ['OPENAI_API_URL']
    api_key = os.environ['OPENAI_API_KEY']
    payload = {
    "model": model,
    "messages": [
      {
        "role": "system",
        "content": f"你的任务是翻译用户提供的srt字幕文件内容为{tgtlang}，请直接返回翻译结果，翻译结果仍然以srt格式给出。注意专有名词不必翻译，保留原样。"
      },
      {
        "role": "user",
        "content": srt
      }
    ]
  }
    r = httpx.post(api_url, json=payload, headers={'Authorization': f'Bearer {api_key}'}, timeout=timeout)
    r.raise_for_status()
    resp = r.json()['choices'][0]['message']['content']
    finish_reason = r.json()['choices'][0]['finish_reason']
    logger.debug(f"{finish_reason=}")
    if resp.startswith('```'):
        resp = '\n'.join(resp.split('\n')[1:])
    if resp.endswith('```'):
        resp = '\n'.join(resp.split('\n')[:-1])
    return resp


def split_srt(srt, max_segments=30):
    segments = Path(srt).read_text(encoding='utf8').split('\n\n')
    chunks = []
    for i in range(0, len(segments), max_segments):
        chunks.append('\n\n'.join(segments[i:i+max_segments]))
    return chunks
