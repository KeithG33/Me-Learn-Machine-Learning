# Playing around more with more reward shaping. This time we make the reward the probability
# of a reward. Thus, being more sure of a move and having it be successful leads to a higher reward.
# On the other hand, failing similarly has a greater "punishment"
# 
import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/RL_animal_kingdom")
from environment_module import *
import pickle  # pickle file for saving/loading prob-tables
import time  # using this to keep track of our saved prob-Tables.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

style.use("ggplot")  
SIZE = 10
NUM_EPISODES = 40_000
SHOW_EVERY = 10_000  # how often to play through enviro visually.
STEPS = 125
NUM_ACTION = 8

start_probs_table = None  # put filename to load from pickle file here.

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
       
       # to find probability of action, a, using beta distribution we calculate
       # alpha / (alpha + beta)
        alpha = probs_table[obs][rand_action][0] + 1
        beta = probs_table[obs][rand_action][1] + 1

        probs = alpha / (alpha + beta)

        if environment.player == environment.pred:
            r_t = 1-probs
            table[obs][rand_action][1] += (1-r_t)          
            end_round = True
        elif environment.player == environment.prey:
            r_t = probs
            table[obs][rand_action][0] += r_t
            end_round = True
        else:
            r_t = 1-probs
            table[obs][rand_action][1] += (1-r_t)

        if end_round:
            environment.reset()
    return table

# Making a probabilities table, gonna be a computing dream. All combinations of pairs (x1,y1),(x2,y2) that make up all possible observations. These fill the Q-table,
# along with Sucess[] and Failure[] arrays for each possible action.
if start_probs_table is None:
    # initialize the q-table
    probs_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        probs_table[((x1, y1), (x2, y2))] = [[1,1] for i in range (8)] # Each obs,action will tally (success,failure)
else:
    with open(start_q_table, "rb") as f: # r for read mode, b for binary...because pickle (b also used for images commonly)
        probs_table = pickle.load(f)

enviro = the_environment() 

probs_table = babble_bae(probs_table,enviro, 1_000_000)
food_count = 0
eaten_count = 0
timeout_count = 0
episode_rewards = []
for episode in range(NUM_EPISODES):

    probabilities = np.zeros(NUM_ACTION)

    # simple no movement no walls enviro for now.
    enviro.reset()

    # If show == true then the enviro will be rendered and we can see what is happening
    if episode % SHOW_EVERY == 0:
            print(f"On EPISODE NUMBER:  {episode}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = False
    else:
        show = False

    episode_reward = 0
    for i in range(STEPS):

        end_episode = False
        
        obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)

        prev_x = enviro.player.x
        prev_y = enviro.player.y

        for action in range(NUM_ACTION):
            probabilities[action] = rnd.beta(probs_table[obs][action][0], probs_table[obs][action][1])
        player_action = np.argmax(probabilities)
        probability = probabilities[player_action]
        enviro.player.action(player_action)

        
        # reward equal to the expectation of reward/penalty that was calculated
        if enviro.player == enviro.pred:
            end_episode = True
            r_t = probability
            probs_table[obs][player_action][1] += 1-r_t
            score = -300
        elif enviro.player == enviro.prey:
            end_episode = True
            r_t = probability
            probs_table[obs][player_action][0] += r_t 
            score = 25
        else:
            r_t = probability
            probs_table[obs][player_action][1] += 1-r_t
            score = -1

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
                       
        episode_reward += score
        if enviro.player == enviro.prey:
            food_count += 1
            break
        elif enviro.player == enviro.pred:
            eaten_count += 1
            break
        elif i == STEPS - 1 and score != 25 and score != -300:
            timeout_count += 1

    #print(episode_reward)
    episode_rewards.append(episode_reward)

print("Got food: {}\n Got eaten: {}\n timeout: {}".format(food_count,eaten_count,timeout_count))

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("Thompson Sampling (Prob Reward) Rolling Average For Each Episode")
plt.show()


#with open(f"probs_table-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(probs_table,f)