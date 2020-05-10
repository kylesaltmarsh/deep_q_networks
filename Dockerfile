FROM python:3.6

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