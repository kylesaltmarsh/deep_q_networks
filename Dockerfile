FROM python:3.6

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install libsm6 libxext6 libxrender-dev ffmpeg

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