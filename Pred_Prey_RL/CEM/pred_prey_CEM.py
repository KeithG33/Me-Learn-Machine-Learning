import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/pred_prey_rl")
from environment_module import the_environment, Creatures
from CEM_policy_net import PolicyNetwork, generate_trials, top_episodes
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image
import heapq
style.use("ggplot")  

SIZE = 10
PLAYER_N = 1  # player key in colour dict  
PREY_N = 2  # prey key in colour dict
PREDATOR_N = 3  # predator key in colour dict
WALL_N = 4 # wall
EMPTY = 5 # the abyss

NUM_ACTIONS = 9
NUM_RUNS = 10_000
STEPS = 125
NUM_TRIALS = 100
TOP_PERCENTILE = 0.8
COMPLETION_SCORE = -20

show_subevery = 4000
SHOW_EVERY = 10_000  # how often to play through enviro visually.

# color dict to label pred/prey/player/obstacle
d = {1: (255, 0, 0),  # player (blue)
     2: (0, 255, 0),  # prey (green)
     3: (0, 0, 255),  # pred (red)
     4: (255,255,255),  # wall (white)
     5: (0, 0, 0)} # the abyss. aka nothing (black)

observation_size = 4 #
hidden_size = 128 # for network

policy_net = PolicyNetwork(observation_size, NUM_ACTIONS, hidden_size) # initialize network
enviro = the_environment() # initialize environment

run_mean_scores = []
for i in range(NUM_RUNS):
    # generate NUM_TRIALS episodes and all their obs, actions, and rewards.
    trial_obs, trial_actions, trial_rewards, trial_avg_score = generate_trials(enviro,policy_net,NUM_TRIALS,STEPS)

    # get the best episodes' obs and actions
    best_ep_obs, best_ep_actions = top_episodes(trial_obs,trial_actions,trial_rewards,TOP_PERCENTILE)

    # train policy by calculating cross entropy loss between the nn's predicted
    # actions and the ones in the best episodes
    policy_net.update_policy(policy_net,best_ep_obs,best_ep_actions)
    run_mean_scores.append(trial_avg_score)
    print("Average trial score is: {}\n Run number is: {}".format(trial_avg_score, i))

plt.plot([i for i in range(len(run_mean_scores))],run_mean_scores)
plt.ylabel("average score 100 trial batch")
plt.xlabel("run #")
plt.title("CEM Rolling Average For Each Episode")
plt.show()


