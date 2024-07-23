FROM ubuntu:22.04

LABEL author="secsilm@outlook.com"

ENV HOME="/root"
ENV TZ=Asia/Shanghai
ENV PATH="$HOME/miniconda/bin:${PATH}"

WORKDIR /app

RUN apt update && apt install xz-utils -y --no-install-recommends
COPY ./ffmpeg-git-arm64-static.tar.xz /app/ffmpeg-git-arm64-static.tar.xz
RUN tar -xvf ./ffmpeg-git-arm64-static.tar.xz && mv ./ffmpeg-git-20240629-arm64-static/ffmpeg ./ffmpeg-git-20240629-arm64-static/ffprobe /usr/local/bin/
COPY ./Miniconda3-latest-Linux-aarch64.sh /app/Miniconda3-latest-Linux-aarch64.sh
RUN bash Miniconda3-latest-Linux-aarch64.sh -b -f -p $HOME/miniconda
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir --disable-pip-version-check

COPY ./ /app

ENTRYPOINT ["/bin/bash", "/app/start.sh"]
