import sys
import argparse
import json
import re
from tqdm.auto import tqdm
import whisper
from pathlib import Path
import subprocess
from utils import ask_llm, split_srt
import time
from loguru import logger


def is_youtube_url(url: str) -> bool:
    """Check if the given string is a YouTube URL."""
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


def download_youtube_video(url: str, output_dir: str = "../data") -> tuple[str, str]:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Base directory to save the downloaded video
        
    Returns:
        Tuple of (path to the downloaded video file, path to the video's directory)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    # First, get the video title to create directory
    title_command = [
        "yt-dlp",
        "--restrict-filenames",
        "--print", "%(title)s",
        url
    ]
    
    logger.info(f"Getting video title for: {url}")
    title_result = subprocess.run(title_command, capture_output=True, text=True)
    
    if title_result.returncode != 0:
        logger.error(f"yt-dlp stderr: {title_result.stderr}")
        raise Exception(f"Failed to get video title. Return code: {title_result.returncode}")
    
    video_title = title_result.stdout.strip().split('\n')[-1]
    
    # Create video-specific directory
    video_dir = output_path / video_title
    if not video_dir.exists():
        video_dir.mkdir(parents=True)
    
    # Download video to the video-specific directory
    # -S "ext" sorts formats by extension preference (mp4 preferred)
    # --restrict-filenames replaces spaces and special chars with underscores
    command = [
        "yt-dlp",
        "-S", "ext",
        "--restrict-filenames",
        "-o", str(video_dir / "%(title)s.%(ext)s"),
        "--print", "after_move:filepath",
        url
    ]
    
    logger.info(f"Downloading YouTube video: {url}")
    logger.info(f"Command: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"yt-dlp stderr: {result.stderr}")
        raise Exception(f"Failed to download YouTube video. Return code: {result.returncode}")
    
    # The last line of stdout contains the downloaded file path
    downloaded_file = result.stdout.strip().split('\n')[-1]
    
    if not Path(downloaded_file).exists():
        raise FileNotFoundError(f"Downloaded file not found: {downloaded_file}")
    
    logger.info(f"Downloaded video to: {downloaded_file}")
    return downloaded_file, str(video_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and embed subtitle to the video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="The video file path or YouTube URL.")
    parser.add_argument("--lang", help="The main language in video.")
    parser.add_argument("--todir", default="../data/", help="The result video dir.")
    parser.add_argument("--stt_model", default="small", help="The STT model.")
    parser.add_argument("--srt", help="The SRT file.")
    parser.add_argument("--tgtlang", default="中文", help="Translate to which lang.")
    parser.add_argument(
        "--trans_model", default="gpt-4o", help="The translation model."
    )
    parser.add_argument(
        "--soft-sub",
        action="store_true",
        help="Use soft subtitles (mov_text) instead of burning them into video. Default is burn-in for better compatibility.",
    )
    parser.add_argument(
        "-M", "--no-merge",
        action="store_true",
        dest="no_merge_segments",
        help="Disable merging of short subtitle segments.",
    )
    parser.add_argument(
        "-w", "--min-words",
        type=int,
        default=3,
        help="Minimum words per segment.",
    )
    parser.add_argument(
        "-d", "--min-dur",
        type=float,
        default=1.0,
        dest="min_duration",
        help="Minimum duration (seconds) per segment.",
    )
    parser.add_argument(
        "-D", "--max-dur",
        type=float,
        default=10.0,
        dest="max_duration",
        help="Maximum duration (seconds) for merged segment.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel workers for translation (1=sequential).",
    )
    parser.add_argument(
        "-F", "--filter",
        action="store_true",
        dest="filter_segments",
        help="Enable filtering of anomalous segments (experimental).",
    )
    parser.add_argument(
        "--no-fix-ts",
        action="store_true",
        dest="no_fix_timestamps",
        help="Disable automatic fixing of anomalous timestamps.",
    )
    return parser.parse_args()


def video2audio(video, todir=None) -> str | Path:
    # ffmpeg -i data/Starship_flight_4.mp4 -q:a 0 -map a out.mp3
    if not Path(video).exists():
        raise FileNotFoundError
    todir = Path(todir if todir else Path(video).parent)
    if not todir.exists():
        todir.mkdir(parents=True)
    tofile = todir / f"{Path(video).stem}.mp3"
    if tofile.exists():
        logger.warning(f"Audio file {tofile} already exists, skipping conversion.")
        return tofile
    command = ["ffmpeg", "-y", "-i", video, "-q:a", "0", "-map", "a", str(tofile)]
    print(command)
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode == 0:
        return tofile
    raise Exception(f"Command failed with return code {result.returncode}.")


def stt(audio, lang=None, todir=None) -> str | Path:
    todir = Path(todir if todir else Path(audio).parent)
    tofile = todir / f"{Path(audio).stem}_transcription.json"
    if tofile.exists():
        logger.warning(
            f"Transcription file {tofile} already exists, skipping transcription."
        )
        return tofile
    result = model.transcribe(str(audio), word_timestamps=False, language=lang)
    # writer = whisper.utils.WriteSRT(todir)  # type: ignore
    # with open(tofile, "w", encoding="utf8") as f:
    #     writer.write_result(result, f)
    with open(tofile, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return tofile


def fix_timestamp_anomalies(transcribe_res_file, max_wps=5.0, min_wps=1.0):
    """
    Fix segments with anomalous timing by adjusting timestamps.
    
    When Whisper assigns wrong timestamps (e.g., speech assigned to silent periods),
    this function adjusts the timing to be more reasonable based on text length.
    
    The key insight: if a segment has very low WPS (words per second), the start time
    is likely wrong. We can estimate a more reasonable start time by:
    - Calculating expected duration based on typical speech rate
    - Moving the start time closer to the end time
    
    Args:
        transcribe_res_file: Path to the transcription JSON file
        max_wps: Maximum expected words per second for normal speech
        min_wps: Minimum expected words per second for normal speech
        
    Returns:
        Path to the fixed transcription JSON file
    """
    tofile = Path(transcribe_res_file).parent / f"{Path(transcribe_res_file).stem}_fixed.json"
    if tofile.exists():
        logger.warning(f"Fixed transcription file {tofile} already exists, skipping fix.")
        return str(tofile)
    
    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)
    
    segments = transcribe_res["segments"]
    if not segments:
        return transcribe_res_file
    
    fixed_segments = []
    fix_count = 0
    
    # Target WPS for estimating correct duration (typical speech rate)
    target_wps = 2.5
    
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        words = len(text.split())
        
        if duration < 0.1 or words == 0:
            fixed_segments.append(seg.copy())
            continue
        
        wps = words / duration
        
        # Check if timing is anomalous (WPS too low = duration too long for text)
        if wps < min_wps and duration > 5:
            # Calculate expected duration based on typical speech rate
            expected_duration = words / target_wps
            expected_duration = max(expected_duration, 1.0)  # At least 1 second
            expected_duration = min(expected_duration, 10.0)  # At most 10 seconds
            
            # Adjust start time (keep end time, move start closer to end)
            new_start = seg["end"] - expected_duration
            
            old_start = seg["start"]
            new_seg = seg.copy()
            new_seg["start"] = new_start
            fixed_segments.append(new_seg)
            
            fix_count += 1
            logger.warning(
                f"Fixed timestamp [{seg['id']}]: {old_start:.1f}-{seg['end']:.1f} -> "
                f"{new_start:.1f}-{seg['end']:.1f} (WPS: {wps:.2f} -> {words/expected_duration:.2f}) "
                f"- \"{text[:40]}...\""
            )
        else:
            fixed_segments.append(seg.copy())
    
    # Create fixed result
    fixed_result = {
        "text": transcribe_res.get("text", ""),
        "segments": fixed_segments,
        "language": transcribe_res.get("language", ""),
    }
    
    with open(tofile, "w", encoding="utf8") as f:
        json.dump(fixed_result, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Fixed {fix_count} segments with anomalous timestamps. Saved to {tofile}")
    return str(tofile)


def filter_anomalous_segments(transcribe_res_file, min_wps=0.8, max_wps=8.0):
    """
    Filter out segments with anomalous timing (likely STT errors or movie dialogue).
    
    This helps remove segments where:
    - Text is assigned to wrong time spans (e.g., speech during movie playback)
    - Duration is way too long/short for the amount of text
    - Likely movie dialogue picked up instead of narrator speech
    
    Args:
        transcribe_res_file: Path to the transcription JSON file
        min_wps: Minimum words per second (below this, timing is suspicious)
        max_wps: Maximum words per second (above this, timing is suspicious)
        
    Returns:
        Path to the filtered transcription JSON file
    """
    tofile = Path(transcribe_res_file).parent / f"{Path(transcribe_res_file).stem}_filtered.json"
    if tofile.exists():
        logger.warning(f"Filtered transcription file {tofile} already exists, skipping filter.")
        return str(tofile)
    
    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)
    
    segments = transcribe_res["segments"]
    if not segments:
        return transcribe_res_file
    
    filtered_segments = []
    removed_count = 0
    
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        words = len(text.split())
        
        if duration < 0.1:
            # Too short duration, skip
            removed_count += 1
            logger.debug(f"Removed segment {seg['id']}: duration too short ({duration:.2f}s)")
            continue
        
        wps = words / duration
        
        # Check for anomalies
        is_anomalous = False
        reason = ""
        
        # Very long duration with few words (likely wrong timing)
        if duration > 15 and words <= 5:
            is_anomalous = True
            reason = f"long duration ({duration:.1f}s) with few words ({words})"
        
        # Very short segments with low WPS - likely movie dialogue picked up
        # These are often single words or short phrases during movie playback
        # Only filter if: duration >= 1.5s, words <= 3, and wps < 0.8
        elif duration >= 1.5 and words <= 3 and wps < 0.8:
            is_anomalous = True
            reason = f"short phrase with low WPS ({wps:.2f}) - likely movie dialogue"
        
        # WPS way too high (duration too short for text)
        elif wps > max_wps and words > 3:
            is_anomalous = True
            reason = f"high WPS ({wps:.2f})"
        
        if is_anomalous:
            removed_count += 1
            logger.warning(f"Removed anomalous segment [{seg['id']}] {seg['start']:.1f}-{seg['end']:.1f}: {reason} - \"{text[:40]}...\"")
        else:
            filtered_segments.append(seg)
    
    # Re-index segments
    for i, seg in enumerate(filtered_segments):
        seg["id"] = i
    
    # Create filtered result
    filtered_result = {
        "text": transcribe_res.get("text", ""),
        "segments": filtered_segments,
        "language": transcribe_res.get("language", ""),
    }
    
    with open(tofile, "w", encoding="utf8") as f:
        json.dump(filtered_result, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Filtered {removed_count} anomalous segments. {len(filtered_segments)} segments remaining. Saved to {tofile}")
    return str(tofile)


def merge_short_segments(transcribe_res_file, min_words=3, min_duration=1.0, max_duration=10.0, max_words=30):
    """
    Merge short subtitle segments into longer, more complete sentences.
    
    This post-processing step combines short segments (like single words or brief phrases)
    into more natural, readable subtitle blocks while respecting timing constraints.
    
    Args:
        transcribe_res_file: Path to the transcription JSON file
        min_words: Minimum number of words per segment (segments shorter than this will be merged)
        min_duration: Minimum duration in seconds for a segment
        max_duration: Maximum duration in seconds for a merged segment
        max_words: Maximum number of words in a merged segment
        
    Returns:
        Path to the merged transcription JSON file
    """
    tofile = Path(transcribe_res_file).parent / f"{Path(transcribe_res_file).stem}_merged.json"
    if tofile.exists():
        logger.warning(f"Merged transcription file {tofile} already exists, skipping merge.")
        return str(tofile)
    
    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)
    
    segments = transcribe_res["segments"]
    if not segments:
        return transcribe_res_file
    
    # Words that indicate an incomplete sentence if they appear at the end
    # These are typically prepositions, articles, conjunctions, possessive pronouns etc.
    incomplete_endings = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'so', 'as', 'at', 'by', 'for', 
        'in', 'of', 'on', 'to', 'with', 'from', 'into', 'like', 'his', 'her', 'its',
        'their', 'my', 'your', 'our', 'that', 'which', 'who', 'whom', 'whose', 'this',
        'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', "it's", "he's", "she's", "we're", "they're", "I'm",
        "you're", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    }
    
    def ends_with_incomplete(text):
        """Check if text ends with a word that suggests an incomplete sentence."""
        text = text.strip()
        if not text:
            return False
        
        # First check: if the text doesn't end with sentence-ending punctuation,
        # it's likely an incomplete sentence (unless it's a very short phrase)
        words = text.split()
        if len(words) >= 3 and not text[-1] in '.!?':
            return True
        
        # Second check: specific words that indicate incomplete sentence
        last_word = words[-1].lower().rstrip('.,!?:;')
        return last_word in incomplete_endings
    
    merged_segments = []
    current_segment = None
    
    for segment in segments:
        text = segment["text"].strip()
        word_count = len(text.split())
        duration = segment["end"] - segment["start"]
        
        if current_segment is None:
            # Start a new segment
            current_segment = {
                "id": len(merged_segments),
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
            }
        else:
            # Check if we should merge with current segment
            current_text = current_segment["text"]
            current_word_count = len(current_text.split())
            current_duration = current_segment["end"] - current_segment["start"]
            potential_duration = segment["end"] - current_segment["start"]
            potential_word_count = current_word_count + word_count
            
            # Decide whether to merge
            should_merge = False
            
            # Merge if current segment is too short (words or duration)
            if current_word_count < min_words or current_duration < min_duration:
                should_merge = True
            
            # Merge if new segment is too short
            if word_count < min_words or duration < min_duration:
                should_merge = True
            
            # Check if current segment ends with an incomplete sentence indicator
            is_incomplete = ends_with_incomplete(current_text)
            if is_incomplete:
                should_merge = True
            
            # Don't merge if result would be too long
            # Use strict limits to keep subtitles readable (aim for ~2 lines max)
            hard_max_duration = 7.0   # Absolute max duration regardless of conditions
            hard_max_words = 20       # Absolute max words regardless of conditions
            
            if is_incomplete:
                # For incomplete sentences, only slightly relaxed limits
                soft_max_duration = min(max_duration * 1.05, hard_max_duration)
                soft_max_words = min(int(max_words * 1.05), hard_max_words)
                if potential_duration > soft_max_duration or potential_word_count > soft_max_words:
                    should_merge = False
                    logger.debug(f"Incomplete sentence not merged due to length: {current_text[:50]}...")
            else:
                if potential_duration > max_duration or potential_word_count > max_words:
                    should_merge = False
            
            # Don't merge if there's a long gap between segments (> 1.5 seconds)
            gap = segment["start"] - current_segment["end"]
            if gap > 1.5:
                should_merge = False
            
            if should_merge:
                # Merge: extend current segment
                current_segment["end"] = segment["end"]
                current_segment["text"] = current_text + " " + text
            else:
                # Finalize current segment and start new one
                merged_segments.append(current_segment)
                current_segment = {
                    "id": len(merged_segments),
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": text,
                }
    
    # Don't forget the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    # Update IDs
    for i, seg in enumerate(merged_segments):
        seg["id"] = i
    
    # Create new transcription result
    merged_result = {
        "text": transcribe_res.get("text", ""),
        "segments": merged_segments,
        "language": transcribe_res.get("language", ""),
    }
    
    with open(tofile, "w", encoding="utf8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Merged {len(segments)} segments into {len(merged_segments)} segments. Saved to {tofile}")
    return str(tofile)


def save_original_srt(transcribe_res_file):
    """Save the original language SRT subtitle file from transcription JSON."""
    tofile = Path(transcribe_res_file).parent / f"{Path(transcribe_res_file).stem}.srt"
    if tofile.exists():
        logger.warning(
            f"Original SRT file {tofile} already exists, skipping generation."
        )
        return str(tofile)

    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)

    segments = []
    for item in transcribe_res["segments"]:
        start = whisper.utils.format_timestamp(
            float(item["start"]), always_include_hours=True, decimal_marker=","
        )
        end = whisper.utils.format_timestamp(
            float(item["end"]), always_include_hours=True, decimal_marker=","
        )
        text = item["text"].strip()
        segments.append(f"{item['id']}\n{start} --> {end}\n{text}")

    srt_content = "\n\n".join(segments)
    tofile.write_text(srt_content, encoding="utf8")
    logger.info(f"Saved original SRT file: {tofile}")
    return str(tofile)


def translate(transcribe_res_file, tgtlang="中文", model="gpt-4o", timeout=5 * 60, max_workers=4):
    tofile = (
        Path(transcribe_res_file).parent
        / f"{Path(transcribe_res_file).stem}_{tgtlang}.srt"
    )
    if tofile.exists():
        logger.warning(
            f"Translated SRT file {tofile} already exists, skipping translation."
        )
        return tofile
    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)
    
    # Build index to span and text mapping
    index2span = {}
    index2text = {}
    for item in transcribe_res["segments"]:
        idx = item["id"]
        index2span[idx] = {
            "start": whisper.utils.format_timestamp(
                float(item["start"]), always_include_hours=True, decimal_marker=","
            ),
            "end": whisper.utils.format_timestamp(
                float(item["end"]), always_include_hours=True, decimal_marker=","
            ),
        }
        index2text[idx] = item["text"].strip()
    
    translated = ask_llm(transcribe_res, tgtlang=tgtlang, model=model, timeout=timeout, max_workers=max_workers)
    
    # Build translation map for quick lookup
    translation_map = {item.index: item.translation.strip() for item in translated.translations}
    
    # Check for missing translations
    all_ids = set(index2span.keys())
    translated_ids = set(translation_map.keys())
    missing_ids = all_ids - translated_ids
    
    if missing_ids:
        logger.warning(f"Missing {len(missing_ids)} translations: {sorted(missing_ids)}")
        logger.warning("Using original text as fallback for missing translations.")
    
    # Generate SRT with all segments, using fallback for missing ones
    segments = []
    for idx in sorted(index2span.keys()):
        span = index2span[idx]
        if idx in translation_map:
            text = translation_map[idx]
        else:
            # Fallback to original text with a marker
            text = f"[未翻译] {index2text[idx]}"
            logger.debug(f"Using fallback for index {idx}: {index2text[idx][:50]}...")
        segments.append(f"{idx}\n{span['start']} --> {span['end']}\n{text}")
    
    srt_content = "\n\n".join(segments)
    tofile.write_text(srt_content, encoding="utf8")
    
    logger.info(f"Translated {len(translated_ids)}/{len(all_ids)} segments, {len(missing_ids)} fallbacks used.")
    return str(tofile)


def split_multi_sentence_subtitle(num, timerange, text, min_duration=5.0, min_chars=8, min_split_duration=2.0):
    """
    Split a subtitle containing multiple sentences into separate subtitles.
    Only splits if conditions are met (conservative approach).
    
    Args:
        min_duration: Minimum total duration to consider splitting
        min_chars: Minimum characters per sentence to split
        min_split_duration: Minimum duration for each part after split
    
    Returns a list of (num, timerange, text) tuples.
    """
    # Parse time range
    start_str, end_str = timerange.split(" --> ")
    
    def parse_time(t):
        h, m, rest = t.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    start = parse_time(start_str)
    end = parse_time(end_str)
    duration = end - start
    
    # Only split if duration is long enough
    if duration < min_duration:
        return [(num, timerange, text)]
    
    # Find sentence boundaries (。!？)
    # Look for sentence-ending punctuation followed by more content
    sentences = []
    current = ""
    for i, char in enumerate(text):
        current += char
        if char in "。！？" and i < len(text) - 1:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    # Only split if we have exactly 2 sentences and both are long enough
    if len(sentences) != 2:
        return [(num, timerange, text)]
    
    if len(sentences[0]) < min_chars or len(sentences[1]) < min_chars:
        return [(num, timerange, text)]
    
    # Allocate time proportionally based on character count
    total_chars = len(sentences[0]) + len(sentences[1])
    ratio = len(sentences[0]) / total_chars
    
    mid_time = start + duration * ratio
    
    # Check if both parts would have enough duration
    first_duration = mid_time - start
    second_duration = end - mid_time
    if first_duration < min_split_duration or second_duration < min_split_duration:
        return [(num, timerange, text)]
    
    # Create two subtitles
    result = [
        (num, f"{start_str} --> {format_time(mid_time)}", sentences[0]),
        (f"{num}b", f"{format_time(mid_time)} --> {end_str}", sentences[1]),
    ]
    
    return result


def adjust_srt(srt, maxlen=28, split_sentences=True):
    blocks = []
    split_count = 0
    for block in Path(srt).read_text(encoding="utf8").split("\n\n"):
        try:
            num, timerange, text = block.split("\n")
        except ValueError as e:
            logger.warning(f"Found malformed block: {block!r}")
            raise e
        
        # Try to split multi-sentence subtitles
        if split_sentences:
            split_results = split_multi_sentence_subtitle(num, timerange, text)
            if len(split_results) > 1:
                split_count += 1
            for sub_num, sub_timerange, sub_text in split_results:
                if len(sub_text) > maxlen:
                    sub_text = "\n".join(
                        [sub_text[i : i + maxlen] for i in range(0, len(sub_text), maxlen)]
                    )
                sub_text = sub_text.strip("。，").strip()
                blocks.append("\n".join([str(sub_num), sub_timerange, sub_text]))
        else:
            if len(text) > maxlen:
                text = "\n".join(
                    [text[i : i + maxlen] for i in range(0, len(text), maxlen)]
                )
            text = text.strip("。，").strip()
            blocks.append("\n".join([num, timerange, text]))
    
    # Re-number all blocks sequentially
    final_blocks = []
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        lines[0] = str(i)
        final_blocks.append("\n".join(lines))
    
    if split_count > 0:
        logger.info(f"Split {split_count} multi-sentence subtitles.")
    
    tofile = Path(srt).parent / f"{Path(srt).stem}_adjusted.srt"
    tofile.write_text("\n\n".join(final_blocks), encoding="utf8")
    return str(tofile)


def merge(video, srt, todir=None, burn_in=True, original_srt=None) -> str | Path:
    """
    Merge video with subtitles.

    Args:
        video: Input video file path
        srt: SRT subtitle file path (translated)
        todir: Output directory
        burn_in: If True, burn subtitles into video (hardcoded, compatible with all players).
                 If False, use soft subtitles (mov_text, may not work on all devices).
        original_srt: Optional original language SRT file to add as second subtitle track
    """
    if not Path(video).exists() or not Path(srt).exists():
        raise FileNotFoundError
    todir = Path(todir if todir else Path(video).parent)
    tofile = todir / f"{Path(video).stem}_sub.mp4"

    if burn_in:
        # Burn-in subtitles (hardcoded) - compatible with all players
        # Need to escape special characters in path for ffmpeg filter
        srt_escaped = (
            str(Path(srt).absolute())
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video,
            "-vf",
            f"subtitles='{srt_escaped}':force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(tofile),
        ]
    else:
        # Soft subtitles (mov_text) - may not work on all devices
        # Add both translated and original subtitles as separate tracks
        if original_srt and Path(original_srt).exists():
            command = [
                "ffmpeg",
                "-y",
                "-i",
                video,
                "-i",
                srt,
                "-i",
                original_srt,
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                "-map",
                "0:v",
                "-map",
                "0:a",
                "-map",
                "1:s",
                "-map",
                "2:s",
                "-metadata:s:s:0",
                "language=chi",
                "-metadata:s:s:0",
                "title=Chinese",
                "-metadata:s:s:1",
                "language=eng",
                "-metadata:s:s:1",
                "title=English",
                str(tofile),
            ]
        else:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                video,
                "-i",
                srt,
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                "-metadata:s:s:0",
                "language=chi",
                str(tofile),
            ]

    print(command)
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode == 0:
        return tofile
    raise Exception(
        f"Command failed with return code {result.returncode}, stderr: {result.stderr}"
    )


def main(
    video,
    lang=None,
    todir=None,
    srt=None,
    tgtlang=True,
    trans_model="gpt-4o",
    burn_in=True,
    merge_segments=True,
    min_words=3,
    min_duration=1.0,
    max_duration=10.0,
    max_workers=4,
    filter_segments=False,
    fix_timestamps=True,
) -> str | Path:
    original_srt = None
    if not srt:
        logger.info(f"Start video2audio")
        s = time.time()
        audio = video2audio(video, todir)
        logger.info(f"Done video2audio, {time.time() - s:.4f} s.")
        logger.info(f"Start stt")
        s = time.time()
        transcribe_res = stt(audio, lang=lang, todir=todir)
        logger.info(f"Done stt, {time.time() - s:.4f} s.")
        # Fix anomalous timestamps
        if fix_timestamps:
            logger.info(f"Start fix_timestamp_anomalies")
            s = time.time()
            transcribe_res = fix_timestamp_anomalies(transcribe_res)
            logger.info(f"Done fix_timestamp_anomalies, {time.time() - s:.4f} s.")
        # Filter anomalous segments (optional)
        if filter_segments:
            logger.info(f"Start filter_anomalous_segments")
            s = time.time()
            transcribe_res = filter_anomalous_segments(transcribe_res)
            logger.info(f"Done filter_anomalous_segments, {time.time() - s:.4f} s.")
        # Merge short segments
        if merge_segments:
            logger.info(f"Start merge_short_segments")
            s = time.time()
            transcribe_res = merge_short_segments(
                transcribe_res,
                min_words=min_words,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            logger.info(f"Done merge_short_segments, {time.time() - s:.4f} s.")
        # Save original language SRT
        logger.info(f"Start save_original_srt")
        s = time.time()
        original_srt = save_original_srt(transcribe_res)
        logger.info(f"Done save_original_srt, {time.time() - s:.4f} s.")
    else:
        transcribe_res = srt
        # Try to find original SRT if it exists
        original_srt_path = Path(srt).parent / f"{Path(srt).stem.replace('_transcription', '')}_transcription.srt"
        if original_srt_path.exists():
            original_srt = str(original_srt_path)
    if tgtlang:
        logger.info(f"Start translate")
        s = time.time()
        srt = translate(transcribe_res, tgtlang=tgtlang, model=trans_model, max_workers=max_workers)
        logger.info(f"Done translate, {time.time() - s:.4f} s.")
    logger.info(f"Start adjust")
    s = time.time()
    srt = adjust_srt(srt)
    logger.info(f"Done adjust, {time.time() - s:.4f} s.")
    logger.info(f"Start merge")
    s = time.time()
    result_video = merge(video, srt, todir=todir, burn_in=burn_in, original_srt=original_srt)
    logger.info(f"Done merge, {time.time() - s:.4f} s.")
    return result_video


if __name__ == "__main__":
    args = parse_args()
    
    # Check if input is a YouTube URL
    video_path = args.video
    todir = args.todir
    if is_youtube_url(args.video):
        logger.info(f"Detected YouTube URL: {args.video}")
        video_path, todir = download_youtube_video(args.video, output_dir=args.todir)
        logger.info(f"YouTube video downloaded to: {video_path}")
        logger.info(f"Working directory: {todir}")
    else:
        # Local video file: create a directory based on video name
        video_file = Path(args.video)
        if video_file.exists():
            video_name = video_file.stem  # filename without extension
            base_dir = Path(args.todir) if args.todir else video_file.parent
            video_dir = base_dir / video_name
            if not video_dir.exists():
                video_dir.mkdir(parents=True)
            todir = str(video_dir)
            logger.info(f"Created working directory for local video: {todir}")
    
    if not args.srt:
        if args.stt_model == "small":
            modelpath = "stt_models/whisper/small.pt"
        elif args.stt_model == "small.en":
            modelpath = "stt_models/whisper/small.en.pt"
        elif args.stt_model == "turbo":
            modelpath = "turbo"
        else:
            modelpath = args.stt_model
        device = None
        if sys.platform == "darwin":
            device = "mps"
        model = whisper.load_model(modelpath, device=device)
        logger.info(
            f"Loaded STT model {args.stt_model} from {modelpath}, device={device}."
        )
    main(
        video_path,
        args.lang,
        todir,
        args.srt,
        args.tgtlang,
        args.trans_model,
        burn_in=not args.soft_sub,
        merge_segments=not args.no_merge_segments,
        min_words=args.min_words,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_workers=args.jobs,
        filter_segments=args.filter_segments,
        fix_timestamps=not args.no_fix_timestamps,
    )
