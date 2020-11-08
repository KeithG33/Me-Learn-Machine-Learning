# Using a 2 linear layer neural network to output probability of reward for each action.
import heapq
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

MOVE_PENALTY = 1  
ENEMY_PENALTY = 300
FOOD_REWARD = 300

GAMMA = 0.9 # reward discount
LEARNING_RATE = 1e-3

MAX_LR = 1e-3
MIN_LR = 3e-3

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=LEARNING_RATE):
        super(PolicyNetwork,self).__init__()

        self.num_actions = num_actions
        # nn.Linear applies a linear transformation to incoming data. The parameters given (num_inputs,hidden_size) give size of (input, output) of layer.
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size,num_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        self.objective = nn.CrossEntropyLoss()

        #  Use this to get a cyclic LR
        #self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=MIN_LR, max_lr=MAX_LR, step_size_up=100,step_size_down=100,cycle_momentum=False)

    # Our input state is passed through linear1 and then RELU activated. Next, this is hucked into linear2
    # and softmaxed to get our output x.
    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x),dim=1)
        return x
    
    # Same thing but no softmax activation so we can use Cross Entropy Loss on the results
    def non_softmax_forward(self,state):
        x = F.relu(self.linear1(state))
        return self.linear2(x)


    def get_action(self,state):
        # turn state into something pytorch can use (tensor).
        state = torch.from_numpy(state).float().unsqueeze(0)

        # do a forward pass to get the action probabilities and then make a random choice from the distribution
        probabilities = self.forward(Variable(state))
        player_action = np.random.choice(self.num_actions,p=np.squeeze(probabilities.detach().numpy())) 

        return player_action
    
    # Doing CEM here so we want to choose the best episodes from our big batch of generated trial episodes.
    # Then we update our policy by using the cross entropy loss between our NN's predicted best action
    # and the action from our best episodes. 
    def update_policy(self, policy_network, obs, actions):
        # make pytorch tensor for obs and action
        obs_tensor = torch.FloatTensor(obs)
        action_tensor = torch.LongTensor(actions)

        # pass obs into NN to see the predicted action "scores" (probabilities).
        action_scores = policy_network.non_softmax_forward(obs_tensor)

        # calculate loss, do a backwards pass to find the gradient, and use this to train the NN
        loss = policy_network.objective(action_scores,action_tensor)
        loss.backward()
        policy_network.optimizer.step()

    # generate a list of observation/action pairs for n = num_of_trials episodes. Save total reward for each episode to train nn
def generate_trials(environment, policy_net, num_of_trials, max_steps):
    episode_observations = []
    episode_actions = []
    episode_total_reward = []
    episode_total_score = []

    score_threshold = False

    for ep_trial in range(num_of_trials):
        environment.reset()
        obs_list = []
        action_list = []
        total_reward = 0
        total_score = 0

        for step in range(max_steps):
            obs = (environment.player-environment.prey, environment.player-environment.pred)

            # flattens an array. <3 Python
            obs = np.asarray([y for x in obs for y in x])
            obs_list.append(obs)

            action = policy_net.get_action(obs)
            environment.player.action(action)
            action_list.append(action)

            # get reward
            if environment.player == environment.pred:
                reward = -ENEMY_PENALTY
                end_round = True
                score = reward
            elif environment.player == environment.prey:
                reward = FOOD_REWARD
                end_round = True
                score = 25
            else:
                reward = -MOVE_PENALTY
                end_round = False
                score = reward
            
            total_reward += reward
            total_score += score
            
            # save the lists of observations and actions taken, and the total reward for each episode
            if end_round:
                episode_observations.append(obs_list)
                episode_actions.append(action_list)
                episode_total_reward.append(total_reward)
                episode_total_score.append(total_score)
                break

    batch_average_score = np.mean(episode_total_score)
    if batch_average_score > 25:
        print(episode_total_score)
    
    return episode_observations, episode_actions, episode_total_reward, batch_average_score

    # want to use the best episodes to learn so we return 
def top_episodes(ep_obs, ep_actions, ep_r_total,percentile):
    top_dawg_obs = []
    top_dawg_actions = []
    # how many best episodes we take
    n = round(len(ep_r_total)*(1-percentile))
    if n < 1:
        n = 1

    # this guy works in linear time to find indices of n biggest elements. By using range(len()) and .take
    # it returns indices instead of the actual highest values
    best_episode_indices = heapq.nlargest(n, range(len(ep_r_total)),ep_r_total.__getitem__)

    for index in best_episode_indices:
        top_dawg_obs.extend(ep_obs[index])
        top_dawg_actions.extend(ep_actions[index])
    return top_dawg_obs, top_dawg_actions