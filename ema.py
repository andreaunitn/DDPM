import torch
import torch.nn as nn
from copy import deepcopy

class EMA(nn.Module):
    def __init__(self, 
                 model, 
                 beta=0.9999, 
                 step_start_ema=2000
                 ):
        
        super(EMA, self).__init__()
        

        self.beta = beta
        self.step_start_ema = step_start_ema
        self.step = 0

        # Create a deepcopy of the model to serve as the "shadow" model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad() = False

    def update_model_average(self, current_model):
        self.step =+ 1

        # If warmup phase just copy the weights exactly
        if self.step < self.step_start_ema:
            self.ema_model.load_state_dict(current_model.state_dict())
            return
        
        # Update parameters (weights + biases)
        for current_params, ma_params in zip(current_model.parameters(), self.ema_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight,  up_weight)

        # Update buffers (e.g. BN statistics) -> copy
        for current_buffers, ma_buffer in zip(current_model.buffers(), self.ema_model.buffers()):
            ma_buffer.data = current_buffers.data

    def update_average(self, old, new):
        if old is None:
            return new
        
        return self.beta * old + (1 - self.beta) * new
    
    def save_checkpoint(self, path):
        torch.save({
            "step": self.step,
            "model_state_dict": self.ema_model.state_dict()
        }, path)

    def load_checkpoint(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.step = checkpoint["step"]
        self.ema_model.load_state_dict(checkpoint["model_state_dict"])