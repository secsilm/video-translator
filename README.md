# Video Translator

上传一个视频，自动加上中文字幕。

## 安装

1. 从[这里](https://www.johnvansickle.com/ffmpeg/)下载 ffmpeg，如果是 M1 mac，选择 `ffmpeg-git-arm64-static.tar.xz`。
2. 下载 miniconda3。
3. `docker build`。

## 使用

```bash
usage: main.py [-h] [--lang LANG] [--todir TODIR] [--stt_model STT_MODEL] [--srt SRT] [--tgtlang TGTLANG] [--trans_model TRANS_MODEL] video

Generate and embed subtitle to the video.

positional arguments:
  video                 The video file.

options:
  -h, --help            show this help message and exit
  --lang LANG           The main language in video. (default: None)
  --todir TODIR         The result video dir. (default: /data/)
  --stt_model STT_MODEL
                        The STT model. (default: small)
  --srt SRT             The SRT file. (default: None)
  --tgtlang TGTLANG     Translate to which lang. (default: 中文)
  --trans_model TRANS_MODEL
                        The translation model. (default: gpt-4o)
```
