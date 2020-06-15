import torch
import torch.utils.data
import torch.nn
import torch.utils.data
import random
import sys
class MatrixFactorization(torch.nn.Module):
    def __init__(self, userSize, numOfSize, latentDim):
        super(MatrixFactorization, self).__init__()
        self.user_factors = torch.nn.Embedding(userSize, latentDim)
        self.item_factors = torch.nn.Embedding(numOfSize, latentDim)
        torch.nn.init.normal_(self.user_factors.weight, std = 0.01)
        torch.nn.init.normal_(self.item_factors.weight, std = 0.01)
    def forward(self, userIds, itemIds):
        return (self.user_factors(userIds) * self.item_factors(itemIds)).sum(dim = -1)
