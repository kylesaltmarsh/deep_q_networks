import gym
import numpy as np
from collections import namedtuple
import collections
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter, init
from torch.nn import functional as F

from tensorboardX import SummaryWriter

import atari_wrappers
from agent import DQNAgent
import utils
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--compute", help="local or sagemaker", default='local', type=str)
args = parser.parse_args()

# Hyperparamters /opt/ml/input/config
if args.compute == 'local':
	f = open(os.getcwd()+'/hyperparameters.json',) 
	hyperparameters = json.load(f) 
	f.close() 
	hyperparameters['path'] = os.getcwd() + "/content/runs"
elif args.compute == 'sagemaker':
	f = open('/opt/ml/input/config/hyperparameters.json',) 
	hyperparameters = json.load(f) 
	f.close()
	hyperparameters['path'] = "/opt/ml/model/"

print('---------------------------------------------------------------------')
print(hyperparameters)
print('---------------------------------------------------------------------')

DQN_HYPERPARAMS = {
	'dueling': False,
	'noisy_net': False,
	'double_DQN': False,
	'n_multi_step': 2,
	'buffer_start_size': 10001,
	'buffer_capacity': 15000,
	'epsilon_start': 1.0,
	'epsilon_decay': 10**5,
	'epsilon_final': 0.02,
	'learning_rate': 5e-5,
	'gamma': 0.99,
	'n_iter_update_target': 1000
}

# could load the local version and cast the types to the passed version

BATCH_SIZE = int(hyperparameters['BATCH_SIZE'])
MAX_N_GAMES = int(hyperparameters['MAX_N_GAMES'])
TEST_FREQUENCY = int(hyperparameters['TEST_FREQUENCY'])

ENV_NAME = "PongNoFrameskip-v4"
SAVE_VIDEO = True
DEVICE = hyperparameters['DEVICE'] # 'cuda' or 'cpu'
SUMMARY_WRITER = True

LOG_DIR = hyperparameters['path'] # 'content/runs' '/opt/ml/model/'
name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

if __name__ == '__main__':

	# create the environment
	env = atari_wrappers.make_env(ENV_NAME)
	if SAVE_VIDEO:
		# save the video of the games
		env = gym.wrappers.Monitor(env, LOG_DIR + "/main-"+ENV_NAME, force=True)
	
	# starting environment state
	obs = env.reset()

	# TensorBoard
	writer = SummaryWriter(log_dir=LOG_DIR+'/'+name + str(time.time())) if SUMMARY_WRITER else None

	# create the agent
	agent = DQNAgent(env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)

	n_games = 0
	n_iter = 0

	# Play MAX_N_GAMES games
	while n_games < MAX_N_GAMES:
		# act greedly
		action = agent.act_eps_greedy(obs)

		# one step on the environment
		new_obs, reward, done, _ = env.step(action)

		# add the environment feedback to the agent
		agent.add_env_feedback(obs, action, new_obs, reward, done)

		# sample and optimize NB: the agent could wait to have enough memories
		agent.sample_and_optimize(BATCH_SIZE)

		obs = new_obs
		if done:
			n_games += 1

			# print info about the agent and reset the stats
			agent.print_info()
			agent.reset_stats()

			if n_games % TEST_FREQUENCY == 0:
				print('Test mean:', utils.test_game(env, agent, 1))

			obs = env.reset()

	writer.close()

# tensorboard --logdir content/runs --host localhost


# TODO kyle
# torch.save(agent.cc.target_nn.state_dict(),'/opt/ml/model/model.pt')

# agent.cc.target_nn.load_state_dict(torch.load('model.pt'))
# agent.cc.target_nn.eval()