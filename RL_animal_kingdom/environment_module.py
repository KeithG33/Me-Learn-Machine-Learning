import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

from PIL import Image

SIZE = 10
PLAYER_N = 1  # player key in colour dict  
PREY_N = 2  # prey key in colour dict
PREDATOR_N = 3  # predator key in colour dict
WALL_N = 4 # wall
EMPTY = 5 # the abyss
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25  

# color dict to label pred/prey/player/obstacle
d = {1: (255, 0, 0),  # player (blue)
     2: (0, 255, 0),  # prey (green)
     3: (0, 0, 255),  # pred (red)
     4: (255,255,255),  # wall (white)
     5: (0, 0, 0)} # the abyss. aka nothing (black) 

HWALL_Y = 5
HWALL_XA =3
HWALL_XB = 7

# Here we describe our environment (a black array of size SIZE) and the creatures in it
class the_environment:
    def __init__(self): 
        self.env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) # initialize to a black array/grid of squares with walls
        ##self.place_horiz_wall(HWALL_XA, HWALL_XB, HWALL_Y)
        self.player = Creatures()
        self.prey = Creatures()
        self.pred = Creatures()
        ##self.wall_a = (HWALL_XA,HWALL_Y)
        ##self.wall_b = (HWALL_XB,HWALL_Y)

        while self.is_occupied(self.player.x, self.player.y):
            self.player = Creatures()
        self.place_creature(self.player.x, self.player.y, PLAYER_N)

        while self.is_occupied(self.prey.x,self.prey.y):
            self.prey = Creatures()
        self.place_creature(self.prey.x, self.prey.y, PREY_N)

        while self.is_occupied(self.pred.x,self.pred.y):
            self.pred = Creatures()
        self.place_creature(self.pred.x, self.pred.y, PREDATOR_N)
    
    def is_occupied(self,x,y):
        if self.env[y][x].all() == 0:
            return False
        else:
            return True

    def reset(self):
        self.env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        self.player = Creatures()
        while self.is_occupied(self.player.x, self.player.y):
            self.player = Creatures()
        self.place_creature(self.player.x, self.player.y, PLAYER_N)

        self.prey = Creatures()
        while self.is_occupied(self.prey.x,self.prey.y):
            self.prey = Creatures()
        self.place_creature(self.prey.x, self.prey.y, PREY_N)

        self.pred = Creatures()
        while self.is_occupied(self.pred.x,self.pred.y):
            self.pred = Creatures()
        self.place_creature(self.pred.x, self.pred.y, PREDATOR_N)


    # No idea why the above code doesnt work when using 255 instead of 0...I'll deal with this later
    def is_wall(self,x,y):
        count = 0
        for value in self.env[y][x]:
            if value == 255:
                count +=1
        if count == 3:
            return True
        else:
            return False

    # method for placing prey/pred/player into the environment. Should have a check maybe, using the is_occupied() method for that tho
    def place_creature(self, x, y, object_type):
        self.env[y][x] = d[object_type]
    
    def remove_creature(self,x,y):
        self.env[y][x] = d[EMPTY]

    # walls will be initialized within the environment first so gonna leave any checking for valid placement until later
    def place_vert_wall(self, start_y, end_y, x):
        for i in range(start_y, end_y):
            self.env[i][x] = d[WALL_N]

    def place_horiz_wall(self, start_x, end_x, y):
        for i in range(start_x, end_x):
            self.env[y][i] = d[WALL_N]
    
    def display_env(self):
        img = Image.fromarray(self.env, 'RGB')
        img = img.resize((500, 500))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(10)
    
    # Checking if the player can see the pred (or vice versa). Need to check if there is a wall in the way by finding
    # angle to walls and player/pred
    def pred_behind_wall(self):
        def find_angle(x1,y1,x2,y2):
            rise = y2-y1
            run = x2-x1
            return np.arctan2(rise,run) * 180 / np.pi + 180 # shifting this makes it between 0->2pi instead of -pi->pi. No negs
        
        x1 = self.player.x
        y1 = self.player.y
        x2 = self.pred.x
        y2 = self.pred.y

        play_wallA_angle = find_angle(x1,y1, HWALL_XA,HWALL_Y)
        play_wallB_angle = find_angle(x1,y1,HWALL_XB,HWALL_Y)
        play_pred_angle = find_angle(x1,y1,x2,y2)
 
        if y1 < HWALL_Y and y2 > HWALL_Y:
            if play_wallB_angle <= play_pred_angle <= play_wallA_angle:
                return True
            else:
                return False
        elif y1 > HWALL_Y and y2 < HWALL_Y:
            if play_wallA_angle <= play_pred_angle <= play_wallB_angle:
                return True
            else:
                return False
        elif y1 == HWALL_Y and y2 != HWALL_Y:
            return False
        elif y1 == HWALL_Y and y2 == HWALL_Y:
            if x1 < HWALL_XA and x2 < HWALL_XA:
                return True
            elif x1 > HWALL_XB and x2 > HWALL_XB:
                return True


class Creatures:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    # define subtraction to get distance for our pred/prey object coords
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    def __eq__(self,other):
        return self.x == other.x and self.y == other.y
    
    # 8 choices for 8 adjacent squares
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=0,y=1)
        elif choice == 5:
            self.move(x=1,y=0)
        elif choice == 6:
            self.move(x=0,y=-1)
        elif choice == 7:
            self.move(x=-1,y=0)
        elif choice == 8:
            self.move(x=0,y=0)

    def move(self, x=False, y=False):
            # If no value for x, move randomly
            if not x:
                self.x += np.random.randint(-1, 2)
            else:
                self.x += x

            # If no value for y, move randomly
            if not y:
                self.y += np.random.randint(-1, 2)
            else:
                self.y += y

            # Doesn't allow x or y to exceed size of map. Keeps value of boundary
            if self.x < 0:
                self.x = 0
            elif self.x > SIZE-1:
                self.x = SIZE-1
            if self.y < 0:
                self.y = 0
            elif self.y > SIZE-1:
                self.y = SIZE-1

    def set_location(self, x, y):
        self.x = x
        self.y = y