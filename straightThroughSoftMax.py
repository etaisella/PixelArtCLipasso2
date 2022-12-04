import torch
from torch import Tensor
import torch.nn as nn

####################################
#### STRAIGHT THROUGH EXPONENT #####
####################################

class ST_exp(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input: Tensor, temp: float):
    out_reg = torch.exp(input)
    out = torch.exp(input * temp)
    ctx.save_for_backward(out_reg)
    return out

  @staticmethod
  def backward(ctx, grad_output):
    out_reg, = ctx.saved_tensors
    grad = out_reg * grad_output
    return grad, None

class ST_SoftMax(nn.Module):
  def __init__(self, temperature):
    super().__init__()
    self.temp = temperature

  def forward(self, x):
    # 1. subtract maximal value for numerical stability
    max_x, _ = torch.max(x, dim=-1, keepdim=True)
    x = x - max_x

    # 2. apply softmax
    exped_x = ST_exp.apply(x, self.temp)
    exped_sum = torch.sum(exped_x, dim=-1, keepdim=True)
    result = exped_x / exped_sum
    return result