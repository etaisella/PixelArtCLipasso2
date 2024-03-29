import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
from torch import Tensor
from unet import UNet, get_noise
from straightThroughSoftMax import ST_SoftMax, StraightThroughSoftMax
from scipy.ndimage.morphology import binary_dilation

###############################
####### CANVAS CLASSES ########
###############################

class Canvas(nn.Module):
  def __init__(self, device,
               palette: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               bg_mask: Tensor=None,
               temperature: float=1.0,
               old_method: bool=False, 
               straight_through=True):
    super().__init__()
    self.has_bg_mask = bg_mask is not None
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w
    weights = torch.empty((self.h, self.w, self.num_colors)).to(device)
    nn.init.xavier_normal_(weights)
    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))

    # if bg_mask is not None, we need to make sure that the background is always white
    if self.has_bg_mask:
      # rescale bg_mask to canvas size
      bg_mask = torch.nn.functional.interpolate(bg_mask.unsqueeze(0).unsqueeze(0), size=(self.h, self.w))
      self.bg_mask = bg_mask.squeeze().bool()
      self.fg_mask = ~self.bg_mask

    self.weight = torch.nn.parameter.Parameter(data=weights, requires_grad=True)
    
    # set other parameters
    self.palette = torch.nn.parameter.Parameter(data=palette, requires_grad=True)
    self.temperature = torch.nn.parameter.Parameter(data=torch.tensor(temperature).to(device), requires_grad=True)
    if old_method:
      self.st_softmax = ST_SoftMax(temperature)
    else:
      self.st_softmax = StraightThroughSoftMax()
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    if self.straight_through:
      norm_weights = self.st_softmax(self.weight)
    else:
      norm_weights = self.softmax(self.weight * self.temperature)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    if self.has_bg_mask:
      colors[:, self.bg_mask] = 1.0
      # make black outline
      fg_mask_numpy = self.fg_mask.cpu().numpy()
      dilated_fg = binary_dilation(fg_mask_numpy, iterations=1)
      outline_mask = torch.from_numpy(dilated_fg ^ fg_mask_numpy).to(self.weight.device)
      colors[:, outline_mask] = 0.0


    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)

    return colors_upscaled, colors

### Canvas Class by Distance: ###

class Canvas_by_Distance(nn.Module):
  def __init__(self, device,
               palette: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0,
               old_method: bool=False,
               straight_through=True):
    super().__init__()
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w
    weights = torch.empty((3, self.h, self.w)).to(device)
    nn.init.xavier_normal_(weights)
    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    self.weight = torch.nn.parameter.Parameter(data=weights, requires_grad=True)
    self.palette = palette
    self.palette_repeated = torch.zeros((self.h, self.w, self.num_colors, 3)).to(device).detach()
    self.palette_repeated[:, :] = self.palette
    self.palette_repeated = torch.permute(self.palette_repeated, (-2, 0, 1, -1))
    self.temperature = temperature
    if old_method:
      self.st_softmax = ST_SoftMax(temperature)
    else:
      self.st_softmax = StraightThroughSoftMax()
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    weight = torch.permute(self.weight, (1, 2, 0))
    weight = torch.sigmoid(weight)
  
    # 1. get diffs
    distances = self.palette_repeated - weight
    # 2. get square and sum to get differences
    distances = torch.sum(distances**2, dim=-1)
    # 3. Arrange for multiplying
    distances = torch.permute(distances, (1, 2, 0))

    if self.straight_through:
      norm_weights = self.st_softmax(distances)
    else:
      norm_weights = self.softmax(distances * self.temperature)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)
    return colors_upscaled

### Canvas Class DIP: ###

class Canvas_No_Palette(nn.Module):
  def __init__(self, device,
               target: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               image_init=True
               ):
    super().__init__()
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))

    # Set UNet backbone
    if canvas_h <= 32 and canvas_w <= 32:
      ratio = 0.25
    elif canvas_h == 224 and canvas_w == 224:
      ratio = 1.0
    
    if image_init:
      self.backbone = UNet(3, 3, ratio).to(device)
      self.net_input = transforms.Resize((int(canvas_h / ratio), int(canvas_w / ratio)))(target.detach())
    else:
      self.backbone = UNet(3, 3, ratio).to(device)
      self.net_input = get_noise(3, 'noise', (int(canvas_h / ratio), int(canvas_w / ratio))).to(device).detach()   
  
  def forward(self) -> Tensor:
    weight = self.backbone(self.net_input)
    weight = torch.squeeze(weight)
    colors = torch.sigmoid(weight)
    ## Smooth output:
    #norm_weights = self.softmax(weight)
    #colors = torch.matmul(norm_weights, self.palette)
    #colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)

    ## PA Output:
    #norm_weights_pa = self.st_softmax(weight)
    #colors_pa = torch.matmul(norm_weights_pa, self.palette)
    #colors_pa = colors_pa.permute(2, 0, 1)
    #colors_upscaled_pa = self.upsample(torch.unsqueeze(colors_pa, 0))
    #colors_upscaled_pa = torch.squeeze(colors_upscaled_pa)

    return colors_upscaled, colors_upscaled

class Canvas_DIP(nn.Module):
  def __init__(self, device,
               palette: Tensor,
               target: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=1.0,
               straight_through=True,
               image_init=True,
               learn_colors=False):
    super().__init__()
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w

    # Set UNet backbone
    if canvas_h <= 32 and canvas_w <= 32:
      ratio = 0.25
    elif canvas_h == 224 and canvas_w == 224:
      ratio = 1.0
    
    if image_init:
      self.backbone = UNet(3, self.num_colors, ratio).to(device)
      self.net_input = transforms.Resize((int(canvas_h / ratio), int(canvas_w / ratio)))(target.detach())
    else:
      self.backbone = UNet(self.num_colors, self.num_colors, ratio).to(device)
      self.net_input = get_noise(self.num_colors, 'noise', (int(canvas_h / ratio), int(canvas_w / ratio))).to(device).detach()   

    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    if learn_colors:
      self.palette = torch.nn.parameter.Parameter(data=palette, requires_grad=True)
    else:
      self.palette = palette
    self.temperature = temperature
    self.st_softmax = StraightThroughSoftMax()
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    weight = self.backbone(self.net_input)
    weight = torch.permute(torch.squeeze(weight), (1, 2, 0))

    # Smooth output:
    norm_weights = self.softmax(weight)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)

    # PA Output:
    norm_weights_pa = self.st_softmax(weight)
    colors_pa = torch.matmul(norm_weights_pa, self.palette)
    colors_pa = colors_pa.permute(2, 0, 1)
    colors_upscaled_pa = self.upsample(torch.unsqueeze(colors_pa, 0))
    colors_upscaled_pa = torch.squeeze(colors_upscaled_pa)

    return colors_upscaled, colors_upscaled_pa

### Canvas Class DIP by distance: ###

class Canvas_DIP_by_distance(nn.Module):
  def __init__(self, device,
               palette: Tensor,
               target: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0,
               old_method: bool=False,
               straight_through=True,
               image_init=True):
    super().__init__()
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w

    # Set UNet backbone
    if canvas_h <= 32 and canvas_w <= 32:
      ratio = 0.25
    elif canvas_h == 224 and canvas_w == 224:
      ratio = 1.0
    
    if image_init:
      self.backbone = UNet(3, self.num_colors, ratio).to(device)
      self.net_input = transforms.Resize((int(canvas_h / ratio), int(canvas_w / ratio)))(target.detach())
    else:
      self.backbone = UNet(self.num_colors, self.num_colors, ratio).to(device)
      self.net_input = get_noise(self.num_colors, 'noise', (int(canvas_h / ratio), int(canvas_w / ratio))).to(device).detach()   

    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    self.palette = palette
    # Set repeated palette
    self.palette_repeated = torch.zeros((self.h, self.w, self.num_colors, 3)).to(device).detach()
    self.palette_repeated[:, :] = self.palette
    self.palette_repeated = torch.permute(self.palette_repeated, (-2, 0, 1, -1))
    self.temperature = temperature
    if old_method:
      self.st_softmax = ST_SoftMax(temperature)
    else:
      self.st_softmax = StraightThroughSoftMax()
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    weight = self.backbone(self.net_input)
    weight = torch.permute(torch.squeeze(weight), (1, 2, 0))
    weight = torch.sigmoid(weight)
  
    # 1. get diffs
    distances = self.palette_repeated - weight
    # 2. get square and sum to get differences
    distances = torch.sum(distances**2, dim=-1)
    # 3. Arrange for multiplying
    distances = torch.permute(distances, (1, 2, 0))

    if self.straight_through:
      norm_weights = self.st_softmax(distances + 1.0)
    else:
      norm_weights = self.softmax(distances * self.temperature)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)
    return colors_upscaled

def canvas_selector(device,
                    palette: Tensor,
                    target: Tensor,
                    use_dip: bool, 
                    straight_through: bool,
                    image_init: bool,
                    by_distance: bool, 
                    canvas_h: int, 
                    canvas_w: int, 
                    temperature: float,
                    old_method: bool=False,
                    no_palette: bool=False
                    ):
  _, _, im_h, im_w = target.size()

  if no_palette:
    print("Selecting: No Palette Canvas")
    canvas = Canvas_No_Palette(device,
                               target,
                               canvas_h,
                               canvas_w,
                               im_h,
                               im_w,
                               image_init=image_init)
    return canvas

  if use_dip:
    if by_distance:
      print("Selecting: DIP by distance")
      canvas = Canvas_DIP_by_distance(device,
                      palette,
                      target, 
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      old_method,
                      straight_through,
                      image_init=image_init)
    else:
      print("Selecting: DIP")
      canvas = Canvas_DIP(device,
                      palette,
                      target, 
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      straight_through,
                      image_init=image_init)
  else:
    if by_distance:
      print("Selecting: No DIP - by distance")
      canvas = Canvas_by_Distance(device, 
                      palette,
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      old_method,
                      straight_through)
    else:
      print("Selecting: No DIP")
      canvas = Canvas(device,
                      palette,
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      old_method,
                      straight_through)
  
  return canvas