import json
from typing import List
import httpx
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
import os
import openai
from pydantic import BaseModel

load_dotenv()

openai_client = openai.OpenAI(
    base_url=os.environ["OPENAI_API_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)


def ask_llm(transcribe_res: dict, tgtlang="中文", model="gpt-4o", timeout=5 * 60):
    subtitles = [
        {"id": seg["id"], "text": seg["text"]} for seg in transcribe_res["segments"]
    ]
    completion = openai_client.beta.chat.completions.parse(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a subtitle translator. Translate each subtitle to Chinese. Do not skip, merge or split lines. ",
            },
            {
                "role": "user",
                "content": json.dumps(subtitles),
            },
        ],
        response_format=TranslationResponse,
    )
    finish_reason = completion.choices[0].finish_reason
    translations = completion.choices[0].message.parsed
    if finish_reason != "stop":
        logger.warning(f"LLM finished with reason: {finish_reason}")
    return translations


def split_srt(srt, max_segments=30):
    segments = Path(srt).read_text(encoding="utf8").split("\n\n")
    chunks = []
    for i in range(0, len(segments), max_segments):
        chunks.append("\n\n".join(segments[i : i + max_segments]))
    return chunks


class SubtitleTranslation(BaseModel):
    index: int
    translation: str


class TranslationResponse(BaseModel):
    translations: List[SubtitleTranslation]
