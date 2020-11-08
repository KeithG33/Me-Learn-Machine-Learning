# On this episode of "The Animal Kingdom" I implement a policy optimization algorithm:
# the policy gradient algorithm known as REINFORCE. Adapting from here: 
# https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
# which has an amazingly detailed derivation as well. 
#
# The REINFORCE algorithm aka Monte Carlo Policy Gradients updates probability distributions of actions
# in order to maximize the objective J = Expectation[sum(r_t+1)]. So higher probabilities for actions
# that have a higher expected reward for a given state, S.
#
# With the objective defined we now have an optimization/maximization problem on our hands, so we can
# perform the gradient ascent using Pytorch.
#
# Notice in this algorithm we are only updating after a full episode is complete, and there is some randomness
# in our action selection. This is what makes it a Monte Carlo rollout method. We perform a full 
# trajectory with one policy, save all the rewards and probabilities, and then update at the end using gradient ascent.

import sys
sys.path.append("/Users/keith/Desktop/Programming/MLML/Pred_Prey_RL")
from environment_module import the_environment, Creatures
from policy_network import PolicyNetwork
import pickle  # pickle file for saving/loading Q-tables
import time  # using this to keep track of our saved Q-Tables.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image

style.use("ggplot")  

SIZE = 10

NUM_ACTIONS = 9
NUM_EPISODES = 40000
STEPS = 125

show_subevery = 40000
CHASE_EVERY = 5
MOVE_PENALTY = -1  
ENEMY_PENALTY = 300
FOOD_REWARD = 300
SHOW_EVERY = 10000  # how often to play through enviro visually.
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

observation_size = 4 # checks signed distance to pred & prey, so (d_pred, d_prey), where d_pred = (x,y), d_prey = (x',y')
hidden_size = 128 # for network
policy_net = PolicyNetwork(observation_size, NUM_ACTIONS, 128)
enviro = the_environment()

episode_scores = []
episode_rewards = []
for episode in range(NUM_EPISODES):
    if episode % 1000 == 0:
        print("episode number is: {}".format(episode))

    # instantiating the enviro
    enviro.reset()

    episode_reward = 0
    log_probs = []
    rewards = []
    scores = []

    # If show == true then the enviro will be rendered and we can see what is happening
    if episode % SHOW_EVERY == 0:
        print(f"on episode: #{episode}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_scores[-SHOW_EVERY:])}")
        show = True
    elif episode % show_subevery == 0:
        show = True
    else:
        show = False


    for steps in range(STEPS):
        obs = (enviro.player-enviro.prey, enviro.player-enviro.pred)
        obs = np.asarray([y for x in obs for y in x]) # flattens previous result into 1-dim array
        
        prev_x = enviro.player.x
        prev_y = enviro.player.y

        # prev_predx = enviro.pred.x
        # prev_predy = enviro.pred.y
        
        action, log_prob = policy_net.get_action(obs) # use our nn to predict best move and get log_probability
        log_probs.append(log_prob)

        enviro.player.action(action)

        # # now make predator chase, comment this out if not wanted
        # if steps % CHASE_EVERY == 0:
        #     enviro.pred_chase(enviro.pred.x,enviro.pred.y)

        # get rewards
        if enviro.player == enviro.pred:
            reward = -ENEMY_PENALTY
            end_round = True
            score = reward
        elif enviro.player == enviro.prey:
            reward = FOOD_REWARD
            end_round = True
            score = 25
        else:
            reward = -MOVE_PENALTY
            end_round = False
            score = reward

        rewards.append(reward)
        scores.append(score)
        if show:
            enviro.remove_creature(prev_x,prev_y)
            
            # if prev_predx == enviro.prey.x and prev_predy == enviro.prey.y:
            #     enviro.remove_creature(prev_predx,prev_predy)
            #     enviro.place_creature(enviro.prey.x,enviro.prey.y,PREY_N)
            # else:
            #     enviro.remove_creature(prev_predx,prev_predy)

            enviro.place_creature(enviro.player.x,enviro.player.y, PLAYER_N)
            enviro.place_creature(enviro.pred.x,enviro.pred.y,PREDATOR_N)
            enviro.display_env()            

            if end_round:
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        if end_round:
            policy_net.update_policy(policy_net, rewards, log_probs)
            total_reward = np.sum(rewards)
            episode_rewards.append(total_reward)
            episode_scores.append(np.sum(scores)) # for comparing to q-learn
            break
            


moving_avg = np.convolve(episode_scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.title("REINFORCE (Cyclic LR)Rolling Average For Each Episode")
plt.show()


#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#    pickle.dump(q_table,f)