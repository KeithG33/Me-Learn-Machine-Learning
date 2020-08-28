# Let the player look ahead n moves into future looking for one with best reward.
# At the moment this is cheating since nothing moves, so its just increasing his sight.
# 
# Ideally I add a model for predicting how the predator moves and using that to make
# this an actual rollout method.
import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/RL_animal_kingdom")
from environment_module import *
import pickle  # pickle file for saving/loading prob-tables
import time  # using this to keep track of our saved prob-Tables.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

style.use("ggplot")  
SIZE = 10
NUM_EPISODES = 40000
SHOW_EVERY = 10000 # how often to play through enviro visually.
EP_STEPS = 125
NUM_ACTION = 8
REWARD_DECAY = 0.98
start_probs_table = None  # put filename to load from pickle file here.

PREY_REWARD = 1500
PRED_PENALTY =1500
MOVE_PENALTY = 10

PLAYER_N = 1  # player key in colour dict  
PREY_N = 2  # prey key in colour dict
PREDATOR_N = 3  # predator key in colour dict
WALL_N = 4 # wall
EMPTY = 5 # the abyss
# color dict to label pred/prey/player/obstacle
d = {1: (255, 0, 0),  # player (blue)
     2: (0, 255, 0),  # prey (green)
     3: (0, 0, 255),  # pred (red)
     4: (255,255,255),  # wall (white)
     5: (0, 0, 0)} # the abyss. aka nothing (black)

rnd = np.random.RandomState(7)

# babble around and pre-train
def babble_bae(table, environment, babble_steps):
    print("Now Babbling...")
    for i in range(babble_steps):

        obs = (environment.player-environment.prey, environment.player-environment.pred)

        end_round = False

        rand_action = np.random.randint(0,8)
        environment.player.action(rand_action)

        if environment.player == environment.pred:
            table[obs][rand_action][1] += PRED_PENALTY
            end_round = True
        elif environment.player == environment.prey:
            table[obs][rand_action][0] += PREY_REWARD
            end_round = True
        else:
            table[obs][rand_action][1] += MOVE_PENALTY 

        if end_round:
            environment.reset()
    return table

# Given n best actions, will go forward m steps and check rewards. Rewards will be tallied and decayed
def forward_look_action_selection(environment, steps, action_list):
    highest_reward = -float('inf')
    start_x = environment.player.x
    start_y = environment.player.y
    reward_totals = []

    for action in action_list:
        total_reward = 0
        player_action = action
        end_round = False

        for i in range(steps):
            decay = 1-i/steps
            # perform actions
            environment.player.action(player_action)
            # get reward
            if environment.player == environment.pred:
                total_reward -= decay * PRED_PENALTY
                end_round = True
            elif environment.player == environment.prey:
                total_reward += decay * PREY_REWARD
                end_round = True
            else:
                total_reward -= decay * MOVE_PENALTY

            if end_round == True:
                environment.player.set_location(start_x,start_y)
                break

            # now make new obs and choose next best step
            obs = (environment.player-environment.prey, environment.player-environment.pred)          
            probabilities = np.zeros(NUM_ACTION)

            for a in range(NUM_ACTION):
                probabilities[a] = rnd.beta(probs_table[obs][a][0], probs_table[obs][a][1]) 
            player_action = np.argmax(probabilities)

        environment.player.x = start_x
        environment.player.y = start_y

        reward_totals.append(total_reward)
    
    # gives indexes of maximum rewards in list, where indexes correspond to actions.
    best_actions_list = np.argwhere(reward_totals == np.max(reward_totals)).flatten().tolist()

    # choose randomly from the best actions in the list, since there will be ties.
    best_action = random.choice(best_actions_list)

    return best_action

# Making a giganto probabilities table. All combinations of pairs (x1,y1),(x2,y2) that make up all possible observations. These fill the table,
# along with Sucess[] and Failure[] arrays for each possible action.
if start_probs_table is None:
    # initialize the probs-table
    probs_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        probs_table[((x1, y1), (x2, y2))] = [[1,1] for i in range (8)] # Each obs,action will tally (success,failure). 
else:
    with open(start_q_table, "rb") as f: # r for read mode, b for binary...because pickle (b also used for images commonly)
        probs_table = pickle.load(f)


food_count = 0
eaten_count = 0
timeout_count = 0
enviro = the_environment() 
probs_table = babble_bae(probs_table,enviro, 5_000)
episode_rewards = []

for episode in range(NUM_EPISODES):

    probabilities = np.zeros(NUM_ACTION)

    # simple no movement no walls enviro for now.
    enviro.reset()

    # If show == true then the enviro will be rendered and we can see what is happening
    if episode % SHOW_EVERY == 0:
            print(f"On EPISODE NUMBER:  {episode}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(EP_STEPS):
        end_episode = False
    
        obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)

        prev_x = enviro.player.x
        prev_y = enviro.player.y
        
        action_list = [action for action in range(NUM_ACTION)]
        
        best_action = forward_look_action_selection(enviro,2,action_list)
        enviro.player.action(best_action)

        if enviro.player == enviro.pred:
            end_episode = True
            probs_table[obs][best_action][1]+=PRED_PENALTY
            reward = -300
        elif enviro.player == enviro.prey:
            end_episode = True
            probs_table[obs][best_action][0]+=PREY_REWARD
            reward = 25
        else:
            probs_table[obs][best_action][1]+=MOVE_PENALTY
            reward = -1

        # Now we need to display the enviro
        if show:
            enviro.remove_creature(prev_x,prev_y)
            enviro.place_creature(enviro.player.x,enviro.player.y, PLAYER_N)
            enviro.display_env()            
            
            if end_episode:
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                       
        episode_reward += reward
        if reward == 25:
            food_count += 1
            break
        elif reward == -300:
            eaten_count += 1
            break
        elif i == EP_STEPS - 1 and reward != 25 and reward != -300:
            timeout_count += 1

    #print(episode_reward)
    episode_rewards.append(episode_reward)
print("Total food count: {} \n Total eaten count: {} \n Timeout count: {}".format(food_count,eaten_count,timeout_count))

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("Thompson Sampling Rolling Average For Each Episode")
plt.show()

#with open("probs_table----------------.pickle", "wb") as f:
#    pickle.dump(probs_table,f)