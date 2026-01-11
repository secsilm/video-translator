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
1. You MUST output EXACTLY {len(subtitles)} translations, one for each input line
2. For each input [ID=X], output with index=X (use the EXACT same ID number)
3. Do NOT skip, merge, or split any lines - translate EVERY single line
4. One input line = one output translation with the SAME ID
5. NEVER combine multiple lines into one translation
6. If an input line is an incomplete sentence fragment (e.g., ends with "like his" or starts with "vice president"), translate ONLY that fragment - do NOT complete the sentence from the next/previous line
7. Preserve the same fragmented structure as the original subtitles

Input format: [ID=number] text
Output: Use that exact number as the index field""",
            },
            {
                "role": "user",
                "content": f"Translate all {len(subtitles)} subtitles. Each [ID=X] must have a translation with index=X. Keep the same fragment structure:\n\n{subtitle_text}",
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
        # Detect and warn about duplicate indices
        seen_indices = {}
        for t in translations.translations:
            if t.index in seen_indices:
                logger.warning(f"Duplicate index {t.index} detected in batch response. First: '{seen_indices[t.index][:30]}...', Second: '{t.translation[:30]}...'")
            else:
                seen_indices[t.index] = t.translation
        
        # Validate: check which expected IDs are present
        returned_ids = set(seen_indices.keys())
        missing_ids = set(expected_ids) - returned_ids
        extra_ids = returned_ids - set(expected_ids)
        
        if extra_ids:
            logger.warning(f"Batch returned unexpected IDs: {sorted(extra_ids)}")
        
        if missing_ids:
            logger.warning(f"Batch missing {len(missing_ids)} IDs: {sorted(missing_ids)}")
        
        if len(translations.translations) != len(expected_ids):
            logger.warning(f"Expected {len(expected_ids)} translations, got {len(translations.translations)}")
            
            # Detect potential index shift: if we got fewer translations and 
            # the content appears to be shifted
            if len(seen_indices) < len(expected_ids) and missing_ids:
                logger.warning(f"Possible index shift detected. Missing: {sorted(missing_ids)}")
        
        # Only return translations that match expected IDs (using first occurrence for duplicates)
        valid_translations = []
        seen_valid = set()
        for t in translations.translations:
            if t.index in expected_ids and t.index not in seen_valid:
                valid_translations.append(t)
                seen_valid.add(t.index)
        
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


def ask_llm(transcribe_res: dict, tgtlang="中文", model="gpt-4o", timeout=5 * 60, batch_size=20, max_workers=4):
    """Translate subtitles in batches with parallel processing.
    
    Args:
        transcribe_res: Transcription result dict with 'segments' key
        tgtlang: Target language
        model: LLM model to use
        timeout: Timeout per request
        batch_size: Number of subtitles per batch
        max_workers: Number of parallel workers (set to 1 for sequential)
    """
    from tqdm.auto import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    subtitles = [
        {"id": seg["id"], "text": seg["text"]} for seg in transcribe_res["segments"]
    ]
    subtitle_map = {s["id"]: s for s in subtitles}
    
    all_translations = {}  # Use dict to avoid duplicates
    
    # Split into batches
    batches = [subtitles[i : i + batch_size] for i in range(0, len(subtitles), batch_size)]
    
    def process_batch(batch_idx, batch):
        """Process a single batch and return results."""
        try:
            result = ask_llm_batch(batch, tgtlang=tgtlang, model=model, timeout=timeout)
            if result and result.translations:
                return [(t.index, t) for t in result.translations]
            else:
                logger.error(f"Failed to translate batch {batch_idx + 1}")
                return []
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with error: {e}")
            return []
    
    # Process batches in parallel
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, i, batch): i 
                for i, batch in enumerate(batches)
            }
            for future in tqdm(as_completed(futures), total=len(batches), desc=f"Translating ({max_workers} workers)"):
                results = future.result()
                for idx, t in results:
                    all_translations[idx] = t
    else:
        # Sequential processing
        for i, batch in enumerate(tqdm(batches, desc="Translating batches")):
            results = process_batch(i, batch)
            for idx, t in results:
                all_translations[idx] = t
    
    # Check for missing translations and retry individually
    # (batch retry may repeat the same error, so use individual translation)
    all_ids = set(subtitle_map.keys())
    translated_ids = set(all_translations.keys())
    missing_ids = all_ids - translated_ids
    
    if missing_ids:
        logger.warning(f"Missing {len(missing_ids)} translations after batch processing: {sorted(missing_ids)}")
        logger.warning(f"Translating missing subtitles individually for accuracy...")
        
        for mid in tqdm(sorted(missing_ids), desc="Translating missing individually"):
            result = translate_single(subtitle_map[mid], tgtlang=tgtlang, model=model)
            if result:
                all_translations[result.index] = result
        
        # Final check
        translated_ids = set(all_translations.keys())
        still_missing = all_ids - translated_ids
        if still_missing:
            logger.error(f"Still missing {len(still_missing)} translations after individual retry: {sorted(still_missing)}")
    
    # Detect and warn about potential content duplication (adjacent lines with similar translations)
    sorted_ids = sorted(all_translations.keys())
    for i in range(len(sorted_ids) - 1):
        curr_id = sorted_ids[i]
        next_id = sorted_ids[i + 1]
        curr_trans = all_translations[curr_id].translation.strip()
        next_trans = all_translations[next_id].translation.strip()
        
        # Check if the current translation ends with content that duplicates the next one
        # This happens when LLM includes the next segment's content in current translation
        if len(curr_trans) > 20 and len(next_trans) > 10:
            # Check if there's significant overlap
            overlap_len = min(len(curr_trans) // 2, len(next_trans))
            if next_trans[:overlap_len] in curr_trans:
                # Found overlap - remove the duplicated part from current translation
                overlap_start = curr_trans.find(next_trans[:overlap_len])
                if overlap_start > 0:
                    fixed_trans = curr_trans[:overlap_start].strip()
                    if len(fixed_trans) > 5:  # Only fix if there's enough content left
                        logger.warning(f"Fixed translation overlap: ID {curr_id} contained content from ID {next_id}")
                        logger.debug(f"  Before: {curr_trans[:50]}...")
                        logger.debug(f"  After: {fixed_trans[:50]}...")
                        all_translations[curr_id] = SubtitleTranslation(index=curr_id, translation=fixed_trans)
    
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
