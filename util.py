import torch
import numpy as np

#a helper class to keep track of the best model during training
class BestModel:
    def __init__(self):
        self.best_model = None
        self.best_loss = float('inf')

    def update(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = model.state_dict()

    def load_best(self, model):
        model.load_state_dict(self.best_model)
        return model