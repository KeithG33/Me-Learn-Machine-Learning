# Using a 2 linear layer neural network to output probability of reward for each action.

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=MIN_LR, max_lr=MAX_LR, step_size_up=100,step_size_down=100,cycle_momentum=False)

    # Our input is a state passed throw linear1 and then RELU activated. Next, this is hucked into linear2
    # and softmaxed to get our output x.
    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x),dim=1)
        return x

    def get_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.forward(Variable(state))

        player_action = np.random.choice(self.num_actions,p=np.squeeze(probabilities.detach().numpy())) 
        log_prob = torch.log(probabilities.squeeze(0)[player_action])

        return player_action, log_prob
    
    def update_policy(self, policy_network, rewards,log_probs):
        discounted_rewards = []
        G_t = 0
        discount_power = 0

        for t in range(len(rewards)):
            # from the formula G_t = sum_t+1=>T(gamma^(t'-t-1)*r_t'). So sum of future discounted rewards 
            G_t = 0
            discount_power=0

            for r in rewards[t:]:
                G_t += (GAMMA**discount_power) * r
                discount_power += 1
            discounted_rewards.append(G_t)
        
        discounted_rewards = torch.tensor(discounted_rewards)

        # now normalize by subtracting the mean and dividing by the standard deviation of all rewards. According
        # to the man Andrej Karpathy himself, "we're always encouraging and discouraging roughly half of the performed actions".
        # This works to control the variance, improve stability.
        #
        # setting the unbiased = False flag means .std() will return 0 for cases where there is a single reward.
        # Returns NaN otherwise...that was fun to figure out.
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(unbiased = False) + 1e-9)

        policy_gradient = []
        for log_prob, G_t in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * G_t)

        # this clears grad for every parameter in optimizer. Good to call before .backwards() otherwise will accumulate
        # gradients from other stuffs
        policy_network.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        
        policy_gradient.backward() # calculate gradient
        policy_network.optimizer.step() # updates value using gradient calculated above
        policy_network.scheduler.step()
