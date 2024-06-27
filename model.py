import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __inti__(self, input_size, outpu_size,hidden_size):
        super().__init__()
        self.liner1 = nn.Linear(input_size, hidden_size)
        self.liner2 = nn.Linear(hidden_size, outpu_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.liner2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state_old, final_move, reward, state_new,done):
        pass