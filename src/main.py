import argparse
from tqdm.auto import tqdm
import whisper
from pathlib import Path
import subprocess
from utils import ask_llm, split_srt
import time
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Generate and embed subtitle to the video.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', help='The video file.')
    parser.add_argument('--lang', help='The main language in video.')
    parser.add_argument('--todir', default='/data/', help='The result video dir.')
    parser.add_argument('--stt_model', default='small', help='The STT model.')
    parser.add_argument('--srt', help='The SRT file.')
    parser.add_argument('--tgtlang', default='中文', help='Translate to which lang.')
    parser.add_argument('--trans_model', default='gpt-4o', help='The translation model.')
    return parser.parse_args()


def video2audio(video, todir=None) -> str|Path:
    # ffmpeg -i data/Starship_flight_4.mp4 -q:a 0 -map a out.mp3
    if not Path(video).exists():
        raise FileNotFoundError
    todir = Path(todir if todir else Path(video).parent)
    if not todir.exists():
        todir.mkdir(parents=True)
    tofile = todir / f"{Path(video).stem}.mp3"
    command = ['ffmpeg', '-y', '-i', video, '-q:a', '0', '-map', 'a', str(tofile)]
    print(command)
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode == 0:
        return tofile
    raise Exception(f"Command failed with return code {result.returncode}.")


def stt(audio, lang=None, todir=None) -> str|Path:
    todir = Path(todir if todir else Path(audio).parent)
    tofile = todir / f"{Path(audio).stem}.srt"
    result = model.transcribe(str(audio), word_timestamps=True, language=lang)
    writer = whisper.utils.WriteSRT(todir)
    with open(tofile, 'w', encoding='utf8') as f:
        writer.write_result(result, f)
    return tofile


def translate(srt, tgtlang='中文', model='gpt-4o', timeout=5*60):
    translated_chunks = []
    for chunk in tqdm(split_srt(srt)):
        translated = ask_llm(chunk, tgtlang=tgtlang, model=model, timeout=timeout)
        translated_chunks.append(translated)
    translated = '\n\n'.join(translated_chunks)
    tofile = Path(srt).parent / f"{Path(srt).stem}_{tgtlang}.srt"
    tofile.write_text(translated, encoding='utf8')
    return str(tofile)


def adjust_srt(srt, maxlen=28):
    blocks = []
    for block in Path(srt).read_text(encoding='utf8').split('\n\n'):
        num, timerange, text = block.split('\n')
        if len(text) > maxlen:
            text = '\n'.join([text[i:i+maxlen] for i in range(0, len(text), maxlen)])
        text = text.strip('。，').strip()
        blocks.append('\n'.join([num, timerange, text]))
    tofile = Path(srt).parent / f"{Path(srt).stem}_adjusted.srt"
    tofile.write_text('\n\n'.join(blocks), encoding='utf8')
    return str(tofile)


def merge(video, srt, todir=None) -> str|Path:
    if not Path(video).exists() or not Path(srt).exists():
        raise FileNotFoundError
    todir = Path(todir if todir else Path(video).parent)
    tofile = todir / f"{Path(video).stem}_sub.mp4"
    command = ['ffmpeg', '-y', '-i', video, '-i', srt, '-c', 'copy', '-c:s', 'mov_text', str(tofile)]
    print(command)
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode == 0:
        return tofile
    raise Exception(f"Command failed with return code {result.returncode}, stderr: {result.stderr}")


def main(video, lang=None, todir=None, srt=None, tgtlang=True, trans_model='gpt-4o') -> str|Path:
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
    result_video = merge(video, srt, todir=todir)
    logger.info(f"Done merge, {time.time() - s:.4f} s.")
    return result_video


if __name__ == '__main__':
    args = parse_args()
    if not args.srt:
        if args.stt_model == 'small':
            modelpath = 'stt_models/whisper/small.pt'
        elif args.stt_model == 'small.en':
            modelpath = 'stt_models/whisper/small.en.pt'
        else:
            raise ValueError(f"Invalid {args.stt_model=}, please choose one from ['small', 'small.en'].")
        model = whisper.load_model(modelpath)
    main(args.video, args.lang, args.todir, args.srt, args.tgtlang, args.trans_model)
