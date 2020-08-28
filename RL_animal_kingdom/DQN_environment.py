import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
from PIL import Image
import time
from modified_tensorboard_class import ModifiedTensorBoard # modifies board to not log after every fit.
import tensorflow as tf

from environment_module import Creatures
SIZE = 10
RETURN_IMAGES = True
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25 
PLAYER_N = 1  # player key in colour dict  
PREY_N = 2  # prey key in colour dict
PREDATOR_N = 3  # predator key in colour dict
WALL_N = 4 # wall
EMPTY = 5 # the abyss
ACTION_SPACE_SIZE = 9
REPLAY_MEMORY_SIZE = 50000
MODEL_NAME = "128x2"


# color dict to label pred/prey/player/obstacle
d = {1: (255, 0, 0),  # player (blue)
     2: (0, 255, 0),  # prey (green)
     3: (0, 0, 255),  # pred (red)
     4: (255,255,255),  # wall (white)
     5: (0, 0, 0)} # the abyss. aka nothing (black) 


class DQN_environ:
    
    def reset(self):
        self.env = np.zeros((SIZE,SIZE,3))
        self.player = Creatures()

        self.prey = Creatures()
        while self.prey == self.player:
            self.prey = Creatures()

        self.pred = Creatures()
        while self.pred == self.prey or self.pred == self.player:
            self.pred = Creatures()

        # # while self.is_occupied(self.player.x, self.player.y):
        # #     self.player = Creatures()
        # # while self.is_occupied(self.prey.x,self.prey.y):
        # #     self.prey = Creatures()
        # # while self.is_occupied(self.pred.x,self.pred.y):
        # #     self.pred = Creatures()

        self.episode_step = 0
        if RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    # will need to make sure action is valid beforehand
    def step(self,action):
        self.episode_step += 1

        self.player.action(action)
        #
        # could add pred/prey moving here
        #
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.prey) + (self.player-self.pred)
        
        if self.player == self.pred:
            reward = -ENEMY_PENALTY
        elif self.player == self.food:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        done = False
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY or self.episode_step >= 200:
            done = True
        
        return new_observation, reward, done
    
    # For CNN
    def get_image(self):
        self.place_creature(self.player.x,self.player.y, PLAYER_N)
        self.place_creature(self.pred.x,self.pred.y,PREDATOR_N)
        self.place_creature(self.prey.x,self.prey.y,PREY_N)
        img = Image.fromarray(self.env,'RGB')
        self.remove_creature(self.player.x,self.player.y)
        self.remove_creature(self.prey.x,self.prey.y)
        self.remove_creature(self.pred.x,self.pred.y)
        return img

    def display_env(self):
        img = self.get_image()
        img = img.resize((500, 500))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(0)

    def place_creature(self, x, y, object_type):
        self.env[y][x] = d[object_type]

    def remove_creature(self,x,y):
        self.env[y][x] = d[EMPTY]


class DQN_Agent:
    def create_model(self):
        conv_model = models.Sequential()

        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu',input_shape=(SIZE,SIZE,3),data_format="channels_last")) 
        conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D(pool_size=(2,2)))
        conv_model.add(layers.Dropout(0.2))

        # conv_model.add(layers.Conv2D(4, (3, 3),activation='relu'))
        # conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # conv_model.add(layers.Dropout(0.2))

        conv_model.add(layers.Flatten())
        conv_model.add(layers.Dense(64))

        conv_model.add(layers.Dense(ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = 9
        conv_model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['acc'])

        return conv_model

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()

        self.target_model.set_weights(self.model.get_weights())

        # Replay memory that holds 1000-50000 actions
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}--{}".format(MODEL_NAME, int(time.time())))

        self.target_model_update_counter = 0
    

    # transition will be a tuple (obs , action, reward, new obs, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    # Uses a minibatch of samples from memory to train agent, if we have more than 1000 memories
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X=[]
        Y=[]

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If the round is over then set reward, otherwise get next max_future_q and calculate actual q
            #
            # In this case we only need what's called the Target Value portion of the Q-learning equation
            # since it is the value we try to converge to and will be used to train our model. For
            # further explanation on target value and where it comes from see TD control algorithms and the
            # Bellman Equation.
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            X.append(current_state)
            Y.append(new_q)

        X_normalized = np.array(X)/255



        # Fit the model on all of the samples
        self.model.fit(X_normalized,np.array(Y),batch_size=MINIBATCH_SIZE,verbose=0,shuffle=False,callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_model_update_counter += 1
        
        if self.target_model_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model_update_counter = 0