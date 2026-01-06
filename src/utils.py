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


def ask_llm_batch(subtitles: list, tgtlang="中文", model="gpt-4o", timeout=5 * 60):
    """Translate a batch of subtitles using LLM."""
    expected_ids = [s['id'] for s in subtitles]
    
    # Include original IDs in the prompt for explicit mapping
    subtitle_lines = []
    for s in subtitles:
        subtitle_lines.append(f"[ID={s['id']}] {s['text']}")
    subtitle_text = "\n".join(subtitle_lines)
    
    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a subtitle translator. Translate each subtitle to {tgtlang}.

CRITICAL RULES:
1. You MUST output EXACTLY {len(subtitles)} translations
2. For each input [ID=X], output with index=X (use the EXACT same ID number)
3. Do NOT skip, merge, or split any lines - translate EVERY single line
4. One input line = one output translation with the SAME ID

Input format: [ID=number] text
Output: Use that exact number as the index field""",
            },
            {
                "role": "user",
                "content": f"Translate all {len(subtitles)} subtitles. Each [ID=X] must have a translation with index=X:\n\n{subtitle_text}",
            },
        ],
        response_format=TranslationResponse,
        timeout=timeout,
    )
    finish_reason = completion.choices[0].finish_reason
    translations = completion.choices[0].message.parsed
    if finish_reason != "stop":
        logger.warning(f"LLM finished with reason: {finish_reason}")
    
    if translations and translations.translations:
        # Validate: check which expected IDs are present
        returned_ids = {t.index for t in translations.translations}
        missing_ids = set(expected_ids) - returned_ids
        
        if missing_ids:
            logger.warning(f"Batch missing {len(missing_ids)} IDs: {sorted(missing_ids)}")
        
        if len(translations.translations) != len(expected_ids):
            logger.warning(f"Expected {len(expected_ids)} translations, got {len(translations.translations)}")
        
        # Only return translations that match expected IDs
        valid_translations = [t for t in translations.translations if t.index in expected_ids]
        return TranslationResponse(translations=valid_translations)
    
    return translations


def translate_single(subtitle: dict, tgtlang="中文", model="gpt-4o", timeout=60):
    """Translate a single subtitle - used as fallback for missing translations."""
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {tgtlang}. Output ONLY the translation, nothing else.",
                },
                {
                    "role": "user",
                    "content": subtitle['text'],
                },
            ],
            timeout=timeout,
        )
        translation = completion.choices[0].message.content.strip()
        return SubtitleTranslation(index=subtitle['id'], translation=translation)
    except Exception as e:
        logger.error(f"Failed to translate single subtitle {subtitle['id']}: {e}")
        return None


def ask_llm(transcribe_res: dict, tgtlang="中文", model="gpt-4o", timeout=5 * 60, batch_size=20, max_retries=3):
    """Translate subtitles in batches to avoid LLM output length limits."""
    from tqdm.auto import tqdm
    
    subtitles = [
        {"id": seg["id"], "text": seg["text"]} for seg in transcribe_res["segments"]
    ]
    subtitle_map = {s["id"]: s for s in subtitles}
    
    all_translations = {}  # Use dict to avoid duplicates
    
    # Process in batches - use smaller batch size for better accuracy
    for i in tqdm(range(0, len(subtitles), batch_size), desc="Translating batches"):
        batch = subtitles[i : i + batch_size]
        batch_result = ask_llm_batch(batch, tgtlang=tgtlang, model=model, timeout=timeout)
        if batch_result and batch_result.translations:
            for t in batch_result.translations:
                all_translations[t.index] = t
        else:
            logger.error(f"Failed to translate batch {i // batch_size + 1}")
    
    # Check for missing translations and retry with batch
    all_ids = set(subtitle_map.keys())
    translated_ids = set(all_translations.keys())
    missing_ids = all_ids - translated_ids
    
    retry_count = 0
    while missing_ids and retry_count < max_retries:
        retry_count += 1
        logger.warning(f"Retry {retry_count}: {len(missing_ids)} missing translations: {sorted(missing_ids)}")
        
        # Retry missing ones in smaller batches
        missing_subtitles = [subtitle_map[id] for id in sorted(missing_ids)]
        retry_batch_size = min(10, len(missing_subtitles))  # Even smaller batch for retries
        
        for i in range(0, len(missing_subtitles), retry_batch_size):
            batch = missing_subtitles[i : i + retry_batch_size]
            batch_result = ask_llm_batch(batch, tgtlang=tgtlang, model=model, timeout=timeout)
            if batch_result and batch_result.translations:
                for t in batch_result.translations:
                    all_translations[t.index] = t
        
        # Recheck missing
        translated_ids = set(all_translations.keys())
        missing_ids = all_ids - translated_ids
    
    # Final fallback: translate missing ones individually
    if missing_ids:
        logger.warning(f"Translating {len(missing_ids)} remaining subtitles individually...")
        for mid in tqdm(sorted(missing_ids), desc="Translating individually"):
            result = translate_single(subtitle_map[mid], tgtlang=tgtlang, model=model)
            if result:
                all_translations[result.index] = result
        
        # Final check
        translated_ids = set(all_translations.keys())
        missing_ids = all_ids - translated_ids
        if missing_ids:
            logger.error(f"Still missing {len(missing_ids)} translations: {sorted(missing_ids)}")
    
    logger.info(f"Translated {len(all_translations)} subtitles out of {len(subtitles)}")
    
    return TranslationResponse(translations=list(all_translations.values()))


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
