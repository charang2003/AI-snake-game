import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        pass
    
    def get_state(self, game):
        pass
    
    def remember(self, state, action, reward, next_state, done):
        pass
    
    def train_long_memory(self):
        pass
    
    def train_short_momory(self):
        pass
    
    def get_action(self, state):
        pass
    
def train():
    pass

if __name__ == '__main__':
    train()