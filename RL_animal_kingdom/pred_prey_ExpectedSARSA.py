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

NUM_EPISODES = 20000
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25  
epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  
SHOW_EVERY = 10000  
STEPS = 125

start_q_table = None  # put filename to load from pickle file here.

LEARNING_RATE = 0.1
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


# Q-table first
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



# Everything is essentially the same as the Q-learning version but for a couple tweaks as we'll see. SARSA is on-policy whereas
# Q-learning is not. What that means is we'll have the same policy for selecting an action and updating our q-table. Theres also a small
# tweak in when actions are taken.

episode_rewards = []
for episode in range(NUM_EPISODES):
    enviro = the_environment() 

    # If show == true then the enviro will be rendered and we can see what is happening
    if episode % SHOW_EVERY == 0:
            print(f"on episode: #{episode}, epsilon is: {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
    else:
        show = False

    # Here is a difference. We choose an action using epsilon-greedy policy before the steps loop begins, and
    # then execute the action within the loop. 
    # 
    obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)
    new_ordered_action_list = np.argsort(q_table[obs])[::-1]

    rando_num = np.random.random()
    if rando_num > epsilon:
        player_action = new_ordered_action_list[0]
    else:
        player_action = np.random.randint(0, 8)
    
    episode_reward = 0

    # Episode step loop
    for i in range(STEPS):

        prev_x = enviro.player.x
        prev_y = enviro.player.y

        # Now we perform our action
        enviro.player.action(player_action)
        index = 0
        while enviro.is_wall(enviro.player.x, enviro.player.y):
            enviro.player.set_location(prev_x, prev_y)
            if rando_num > epsilon:
                index+=1
                player_action = new_ordered_action_list[index]
            else:
                player_action = np.random.randint(0,8)

            enviro.player.action(player_action)

        # Rewards
        if enviro.player.x == enviro.pred.x and enviro.player.y == enviro.pred.y:
            reward = -ENEMY_PENALTY
        elif enviro.player.x == enviro.prey.x and enviro.player.y == enviro.prey.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
    

        # Instead of choosing argmax(Q) like in Q-learning, SARSA applies the same epsilon-greedy policy as before
        # to calculate Q(S',A') where S' = State_t+1, A' = Action_t+1.

        new_obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)
        new_ordered_action_list = np.argsort(q_table[new_obs])[::-1]

        curr_x = enviro.player.x
        curr_y = enviro.player.y

        rando_num = np.random.random()
        if rando_num > epsilon:
            i=0
            new_player_action = new_ordered_action_list[i]
            enviro.player.action(new_player_action)

            # This no bueno since it could theoretically get an out of bounds error. Lets just assume for now there won't be a wall in every direction
            while enviro.is_wall(enviro.player.x, enviro.player.y):
                enviro.player.set_location(prev_x, prev_y)
                i+=1
                new_player_action = new_ordered_action_list[i]
                enviro.player.action(new_player_action)
        else:
            new_player_action = np.random.randint(0, 8)
            enviro.player.action(new_player_action)

            while enviro.is_wall(enviro.player.x,enviro.player.y):
                enviro.player.set_location(prev_x, prev_y)
                new_player_action = np.random.randint(0, 8)
                enviro.player.action(new_player_action)

        # putting player back, only executed the action above so I could check if valid (wall or not)        
        enviro.player.set_location(curr_x,curr_y)

        current_q = q_table[obs][player_action]
        next_expectation = np.mean(q_table[new_obs][:])

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * next_expectation)
        q_table[obs][player_action] = new_q

        # After this is done now S <-- S' and A <-- A' sooo...
        player_action = new_player_action


        if show:
            enviro.remove_creature(prev_x,prev_y)
            enviro.place_creature(enviro.player.x,enviro.player.y, PLAYER_N)

            enviro.display_env()            
   
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD:
            print("Food caught :D :D....We done it!!!")
            break
        elif reward == -ENEMY_PENALTY:
            print("Eaten by a predator :( :(")
            break
    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("Expected-SARSA Rolling Average For Each Episode")
plt.show()

#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table,f)
