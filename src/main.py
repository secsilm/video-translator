import sys
import argparse
import json
from tqdm.auto import tqdm
import whisper
from pathlib import Path
import subprocess
from utils import ask_llm, split_srt
import time
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and embed subtitle to the video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="The video file.")
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


def translate(transcribe_res_file, tgtlang="中文", model="gpt-4o", timeout=5 * 60):
    tofile = (
        Path(transcribe_res_file).parent
        / f"{Path(transcribe_res_file).stem}_{tgtlang}.srt"
    )
    if tofile.exists():
        logger.warning(
            f"Translated SRT file {tofile} already exists, skipping translation."
        )
        return tofile
    # translated_chunks = []
    # for chunk in tqdm(split_srt(srt)):
    #     translated = ask_llm(chunk, tgtlang=tgtlang, model=model, timeout=timeout)
    #     translated_chunks.append(translated)
    # translated = "\n\n".join(translated_chunks)
    with open(transcribe_res_file, "r", encoding="utf8") as f:
        transcribe_res = json.load(f)
    index2span = {
        item["id"]: {
            "start": whisper.utils.format_timestamp(
                float(item["start"]), always_include_hours=True, decimal_marker=","
            ),
            "end": whisper.utils.format_timestamp(
                float(item["end"]), always_include_hours=True, decimal_marker=","
            ),
        }
        for item in transcribe_res["segments"]
    }
    translated = ask_llm(transcribe_res, tgtlang=tgtlang, model=model, timeout=timeout)
    # translated = [(item.index, item.translation) for item in translated.translations]
    segments = []
    for item in translated.translations:
        if item.index not in index2span:
            logger.warning(f"Index {item.index} not found in transcribe_res.")
            continue
        span = index2span[item.index]
        segments.append(
            f"{item.index}\n{span['start']} --> {span['end']}\n{item.translation.strip()}"
        )
    srt_content = "\n\n".join(segments)
    tofile.write_text(srt_content, encoding="utf8")
    return str(tofile)


def adjust_srt(srt, maxlen=28):
    blocks = []
    for block in Path(srt).read_text(encoding="utf8").split("\n\n"):
        try:
            num, timerange, text = block.split("\n")
        except ValueError as e:
            logger.warning(f"Found malformed block: {block!r}")
            raise e
        if len(text) > maxlen:
            text = "\n".join(
                [text[i : i + maxlen] for i in range(0, len(text), maxlen)]
            )
        text = text.strip("。，").strip()
        blocks.append("\n".join([num, timerange, text]))
    tofile = Path(srt).parent / f"{Path(srt).stem}_adjusted.srt"
    tofile.write_text("\n\n".join(blocks), encoding="utf8")
    return str(tofile)


def merge(video, srt, todir=None, burn_in=True) -> str | Path:
    """
    Merge video with subtitles.

    Args:
        video: Input video file path
        srt: SRT subtitle file path
        todir: Output directory
        burn_in: If True, burn subtitles into video (hardcoded, compatible with all players).
                 If False, use soft subtitles (mov_text, may not work on all devices).
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
) -> str | Path:
    if not srt:
        logger.info(f"Start video2audio")
        s = time.time()
        audio = video2audio(video, todir)
        logger.info(f"Done video2audio, {time.time() - s:.4f} s.")
        logger.info(f"Start stt")
        s = time.time()
        srt = stt(audio, lang=lang, todir=todir)
        logger.info(f"Done stt, {time.time() - s:.4f} s.")
    if tgtlang:
        logger.info(f"Start translate")
        s = time.time()
        srt = translate(srt, tgtlang=tgtlang, model=trans_model)
        logger.info(f"Done translate, {time.time() - s:.4f} s.")
    logger.info(f"Start adjust")
    s = time.time()
    srt = adjust_srt(srt)
    logger.info(f"Done adjust, {time.time() - s:.4f} s.")
    logger.info(f"Start merge")
    s = time.time()
    result_video = merge(video, srt, todir=todir, burn_in=burn_in)
    logger.info(f"Done merge, {time.time() - s:.4f} s.")
    return result_video


if __name__ == "__main__":
    args = parse_args()
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
        args.video,
        args.lang,
        args.todir,
        args.srt,
        args.tgtlang,
        args.trans_model,
        burn_in=not args.soft_sub,
    )
