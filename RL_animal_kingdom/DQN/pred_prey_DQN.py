from environment_module import Creatures
from DQN_environment import DQN_environ, DQN_Agent
from keras import models
from keras import layers
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np
import time
import random
from tqdm import tqdm
import os
import cv2
from PIL import Image
from modified_tensorboard_class import ModifiedTensorBoard # modifies board to not log after every fit.

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "128x2"
UPDATE_TARGET_EVERY = 5
DISCOUNT=0.99
MINIBATCH_SIZE = 64
MIN_REWARD = -200
MEMORY_FRACTION = 0.2

EPISODES = 20000

epsilon = 1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50 #episodes
SHOW_PREVIEW = False

ACTION_SPACE_SIZE = 9
SIZE = 10

ep_reward = [-200]

# Allows us to reproduce numbers for testing, sets pseudo-RNG seed.
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

env = DQN_environ()
agent = DQN_Agent()

# Episode loop..tqdm is for a progress bar, fancy fancy
for episode in tqdm(range(1,EPISODES+1),ascii=True, unit='episodes'):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # e-greedy with a CNN agent giving us q values instead of a q_table
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state,action,reward,new_state,done))
        agent.train(done, step)
        current_state = new_state
        step += 1

 # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)