# In this Thompson Sampling Algorithm I'm choosing actions by finding the one with the highest
# expectation of reward. Each (state,action) pair has a beta-distribution that quantifies the
# expectation/uncertainty of reward. Thus, we sample from these distributions and choose
# the best action. Since these are distributions and we're randomly sampling, the possibility
# still exists for exploring what we think are bad moves.
# 
# In practice the b-distributions are updated using Bayes Rule with
# their two parameters alpha and beta according to: a += r_t, b += 1 - r_t, where r_t is reward for a given time, t.
#
#  Note in the above example, if reward is 1 then 'a' shifts up one and 'b' stays the same. Similarly, if 0, then
# 'b' goes up 1 and 'a' stays the same. The relevance of this (how it changes the distribution) is clear when 
# looking at pics of how beta distributions change with their parameters...and is pretty badass cool to use
# like this, dare I say so. Anyway, I took that idea for this implementation. When food is found I shift 'a' up by
# PREY_REWARD. When I'm eaten or move 'b' goes down by PRED_PENALTY, MOVE_PENALTY.
# 
import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/RL_animal_kingdom")
from environment_module import *
import pickle  # pickle file for saving/loading prob-tables
import time  # using this to keep track of our saved prob-tables.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

style.use("ggplot")  
SIZE = 10
NUM_EPISODES = 40000
SHOW_EVERY = 10000  # how often to play through enviro visually.
STEPS = 125
NUM_ACTION = 8

start_probs_table = None  # put filename to load from pickle file here.

PREY_REWARD = 15000
PRED_PENALTY = 15000
MOVE_PENALTY = 100

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


# Making a giant probabilities table, a computing dream. All combinations of pairs (x1,y1),(x2,y2) that make up all possible observations. These fill the Q-table,
# along with Sucess[] and Failure[] arrays for each possible action.
if start_probs_table is None:
    # initialize the q-table
    probs_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        probs_table[((x1, y1), (x2, y2))] = [[0,0] for i in range (8)] # Each obs,action will tally (success,failure)
else:
    with open(start_q_table, "rb") as f: # r for read mode, b for binary...because pickle (b also used for images commonly)
        probs_table = pickle.load(f)

enviro = the_environment()

probs_table = babble_bae(probs_table,enviro, 50_000)
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
        
        # Use a beta distribution to model our expectation of reward. Here we randomly sample probabilities for reward and
        # choose the action with the highest.
        #
        # By randomly sampling from a beta distribution (TS is also called posterior sampling), we have the chance of exploring
        # actions we are uncertain about in order to learn. Beta distributions also have the unique property that updating
        # bayes rule consists of updating the alpha and beta parameters. Once we update our posterior enough our alpha parameter
        # will increase making it more certain of reward, thus less likely to explore. This is the beauty of Thompson Sampling, it's
        # an excellent algorithm to balance exploration vs exploitation.

        for action in range(NUM_ACTION):
            probabilities[action] = rnd.beta(probs_table[obs][action][0] + 1, probs_table[obs][action][1] + 1) # starts at (a,b)=(1,1) since a,b =/= 0
        player_action = np.argmax(probabilities)
        enviro.player.action(player_action)

        if enviro.player == enviro.pred:
            end_episode = True
            probs_table[obs][player_action][1]+=PRED_PENALTY
            reward = -300
        elif enviro.player == enviro.prey:
            end_episode = True
            probs_table[obs][player_action][0]+=PREY_REWARD
            reward = 25
        else:
            probs_table[obs][player_action][1]+=MOVE_PENALTY
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
            #print("Food caught :D :D....We done it!!!")
            break
        elif reward == -300:
            #print("Eaten by a predator :( :(")
            break
        elif i == STEPS - 1 and reward != 25 and reward != -300:
            #print("Timed out the steps...")
            pass

    #print(episode_reward)
    episode_rewards.append(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("Thompson Sampling Rolling Average For Each Episode")
plt.show()


#with open("probs_table----------------.pickle", "wb") as f:
#    pickle.dump(probs_table,f)