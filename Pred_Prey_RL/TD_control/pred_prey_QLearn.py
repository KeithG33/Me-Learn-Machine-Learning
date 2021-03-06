# This began as a simple Q-learning script from watching a youtube video by pythonprogamming.net. Credit to them
# for the skeleton of this code, and many thanks for the fun toy environment/problem to play with.
#
# I've moved the classes into a module, added walls to the environment, and given the predator the ability to
# chase the player, which will interesting when I add a model to predict these movements
import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/RL_animal_kingdom")
from environment_module import *
import pickle  # pickle file for saving/loading Q-tables
import time  # using this to keep track of our saved Q-Tables.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image


style.use("ggplot")  

SIZE = 10

NUM_EPISODES = 40000
MOVE_PENALTY = 1  
ENEMY_PENALTY = 3000  
FOOD_REWARD = 3000 
epsilon = 0.5  # randomness
EPS_DECAY = 0.998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000  # how often to play through enviro visually.
STEPS = 125

start_q_table = None  # put filename to load from pickle file here.

LEARNING_RATE = 0.15
DISCOUNT = 0.95

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

food_count = 0
pred_eaten = 0
# Now the Q-Learning begins. Need Q-table first
#
# Here we have "for four-loops fore" [sic] all combinations of pairs (x1,y1),(x2,y2) that make up all possible observations. These fill the Q-table,
# along with values for each possible action. Yucky but meh for now.
if start_q_table is None:
    # initialize the q-table
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range (8)]
else:
    with open(start_q_table, "rb") as f: # r for read mode, b for binary...because pickle (b also used for images commonly)
        q_table = pickle.load(f)


# Finally the magic happens. Each outer loop is an episode, with STEP number of steps within that. Each step will...
# make an obversation --> choose an action --> get reward --> update table. Classic Q-Learning.
episode_rewards = []
episode_scores = []
for episode in range(NUM_EPISODES):

    # instantiating the enviro will create an enviro with creatures and wall in random positions
    enviro = the_environment() 

    # If show == true then the enviro will be rendered and we can see what is happening
    if episode % SHOW_EVERY == 0:
            print(f"on episode: #{episode}, epsilon is: {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_scores[-SHOW_EVERY:])}")
            show = True
    else:
        show = False
    episode_score = 0
    episode_reward = 0
    for i in range(STEPS):
        obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)

        # descending list for q_table[obs] so we can go through actions in the case of non-valid actions
        ordered_action_list = np.argsort(q_table[obs])[::-1]

        prev_x = enviro.player.x
        prev_y = enviro.player.y

        # # # prev_predx = enviro.pred.x
        # # # prev_predy = enviro.pred.y
 
        # Epsilon-greedy policy to choose and take an action.
        if np.random.random() > epsilon:
            i=0
            player_action = ordered_action_list[i]
            enviro.player.action(player_action)

            # Choose the next best move until we aren't moving into a wall
            while enviro.is_wall(enviro.player.x, enviro.player.y):
                enviro.player.set_location(prev_x, prev_y)
                i+=1
                player_action = ordered_action_list[i]
                enviro.player.action(player_action)
        else:
            player_action = np.random.randint(0, 8)
            enviro.player.action(player_action)
            
            while enviro.is_wall(enviro.player.x,enviro.player.y):
                enviro.player.set_location(prev_x, prev_y)
                player_action = np.random.randint(0, 8)
                enviro.player.action(player_action)

        #### LATER ###
        # enviro.prey.move()
        # enviro.pred_chase()
        #### LATER ###         

        # Set rewards for getting prey or being eaten by predator, otherwise moves get small penalty and eating our prey is big win-win.
        if enviro.player == enviro.pred:
            reward = -ENEMY_PENALTY
            score = reward
        elif enviro.player == enviro.prey:
            reward = FOOD_REWARD
            score = 25
        else:
            reward = -MOVE_PENALTY
            score = reward

        # now make new obs immediately after the move and do our calcs for that magical Q-learning formula. I love humans...
        new_obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][player_action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][player_action] = new_q

        # The Q-learning part is now done after using the above formula and updating the Q-values in the table.
        #
        # Now we need to display the enviro
        if show:
            enviro.remove_creature(prev_x,prev_y)
            enviro.place_creature(enviro.player.x,enviro.player.y, PLAYER_N)
  
            # # # enviro.remove_creature(prev_predx,prev_predy)
            # # # if prev_predx == enviro.prey.x and prev_predy == enviro.prey.y:
            # # #     enviro.place_creature(enviro.prey.x,enviro.prey.y, PREY_N)
            # # # enviro.place_creature(enviro.pred.x,enviro.pred.y, PREDATOR_N)

            enviro.display_env()            
            # If the round is over we want to see the ending to determine if our player failed/succeeded. In the OpenCV documentation apparently
            # it's a general convention to use "q" for halting indefinite operations. Now I know hehe.
            #
            # Anyway...WaitKey returns 32bit integer of pressed key, ord('q') returns unicode code pointer of q and the 0xFF is a bit mask
            # just like we did in CPEN 312. Masks the MSD's making them 0, leaving the LSD digits we care about for the comparison. 
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        episode_score += score
        episode_reward += reward
        if reward == FOOD_REWARD:
            break
        elif reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_scores.append(episode_score)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("Q-Learn Rolling Average For Each Episode")
plt.show()


#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table,f)