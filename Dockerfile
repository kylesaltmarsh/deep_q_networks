# FROM python:3.6
FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install ffmpeg

# set the working directory
RUN ["mkdir", "deep_q_networks"]
WORKDIR "deep_q_networks"

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

# install environment dependencies
COPY "run.sh" .
COPY "main.py" .
COPY "central_control.py" .
COPY "buffers.py" .
COPY "agent.py" .
COPY "atari_wrappers.py" .
COPY "neural_net.py" .
COPY "utils.py" .

# provision environment
RUN ["chmod", "+x", "./run.sh"]
ENTRYPOINT ["./run.sh"]
CMD ["train"]