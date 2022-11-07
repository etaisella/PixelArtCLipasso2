import torch
import collections
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple
from torch import Tensor
from PIL import Image
import click
import clip
import os
from easydict import EasyDict

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# set device and load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# CLIP Setup
print(f"Available CLiP models - {clip.available_models()}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# clip preprocess for tensor
def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

clip_tensor_preprocess = clip_transform(224)
to_PIL = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# -------------------------------------------------------------------------------------
#  Command line configuration for the script                                          |
# -------------------------------------------------------------------------------------
# fmt: off
# noinspection PyUnresolvedReferences
@click.command()
## Required arguments:
@click.option("-i", "--input_image", type=click.Path(file_okay=True, dir_okay=False),
              required=True, help="path to the target image")

# Non-required Configurations options:
# Pipeline
@click.option("--use_dip", type=click.BOOL, required=False, default=True,
              help="whether to use Deep Image Priors in the pipeline",
              show_default=True)
@click.option("--init_image", type=click.BOOL, required=False, default=False,
              help="whether to init the net input to the target image",
              show_default=True)
@click.option("--straight_through", type=click.BOOL, required=False, default=False,
              help="whether to use the straight through softmax",
              show_default=True)
@click.option("--learn_colors", type=click.BOOL, required=False, default=False,
              help="whether to learn the color palette over training",
              show_default=True)
@click.option("--by_distance", type=click.BOOL, required=False, default=False,
              help="whether to init the net input to the target image",
              show_default=True)
@click.option("--temperature", type=click.FLOAT, required=False, default=1000.0,
              help="softmax temperature",
              show_default=True)
@click.option("--canvas_h", type=click.INT, required=False, default=32,
              help="canvas height",
              show_default=True)
@click.option("--canvas_w", type=click.INT, required=False, default=32,
              help="canvas width",
              show_default=True)
@click.option("--num_colors", type=click.INT, required=False, default=6,
              help="number of colors in palette",
              show_default=True)

# Losses
@click.option("--l2_weight", type=click.FLOAT, required=False, default=1.0,
              help="l2 weight", show_default=True)
@click.option("--style_weight", type=click.FLOAT, required=False, default=0.0,
              help="style weight", show_default=True)
@click.option("--style_prompt", type=click.STRING, required=False, default="pixel art",
              help="style input prompt", show_default=True)
@click.option("--semantic_weight", type=click.FLOAT, required=False, default=0.0,
              help="semantic weight", show_default=True)
@click.option("--geometric_weight", type=click.FLOAT, required=False, default=0.0,
              help="geometric weight", show_default=True)

# Training
@click.option("--lr", type=click.FLOAT, required=False, default=0.0005,
              help="learning rate", show_default=True)
@click.option("--save_freq", type=click.INT, required=False, default=100,
              help="frequency to save results in", show_default=True)
@click.option("--epochs", type=click.INT, required=False, default=1000,
              help="number of epochs", show_default=True)


# fmt: on
# -------------------------------------------------------------------------------------

def main(**kwargs) -> None:
    # load the requested configuration for the training
    config = EasyDict(kwargs)

    # get image and palette
    image_path = "inputs" / Path(config.input_image)
    target, palette = get_target_and_palette(image_path, config.num_colors)

    # get canvas class
    canvas = canvas_selector(palette, target, config.use_dip, config.straight_through,
                            config.init_image, config.by_distance, config.canvas_h, config.canvas_w, 
                            config.temperature)
    
    # set losses
    loss_dict = dict.fromkeys(["l2", "semantic", "style", "geometric"], 
                          torch.tensor([0.0]).to(device))
    loss_dict["semantic"] = torch.tensor([config.semantic_weight]).to(device)
    loss_dict["style"] = torch.tensor([config.style_weight]).to(device)
    loss_dict["geometric"] = torch.tensor([config.geometric_weight]).to(device)
    loss_dict['l2'] = torch.tensor([config.l2_weight]).to(device)
    clip_conv_layer_weights = [0, 0.0, 1.0, 1.0, 0]

    loss_fn = PixelArtLoss(clip_model,
                           target,
                           loss_dict,
                           clip_conv_layer_weights,
                           config.style_prompt)

    # set optimizer
    optimizer = torch.optim.Adam(canvas.parameters(), lr=config.lr)

    #----------------#
    # Training Loop:
    #----------------#

    checkpoints = []
    for iter in range(config.epochs):
      optimizer.zero_grad()
      output = canvas()
      loss = loss_fn(output)
      loss.backward()
      optimizer.step()
      if (iter % config.save_freq == 0):
        checkpoint = {}
        checkpoint["iteration"] = iter
        checkpoint["loss"] = loss.item()
        print(f"Iter: {iter}, Loss: {loss.item()}")
        checkpoint["frame"] = torch.squeeze(output).cpu().detach()
        checkpoints.append(checkpoint)

    # Save Results:
    plot_results(checkpoints, config)

####################################
####     PLOTTING FUNCTIONS    #####
####################################

def to8b(x: np.array) -> np.array:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def plot_results(checkpoints, config):
  losses = [x['loss'] for x in checkpoints]
  tmp = min(losses)
  idx = losses.index(tmp)
  frame = checkpoints[idx]['frame']
  frame_pil = to_PIL(frame)
  title = f"{config.num_colors} colors \n l2 {config.l2_weight}, semantic {config.semantic_weight},geometric {config.geometric_weight}, style {config.style_weight}\n use dip {config.use_dip},straight through {config.straight_through} \nby distance {config.by_distance} \nstyle prompt - {config.style_prompt}"
  filename = f"{config.num_colors}_colors_l2_{config.l2_weight}_semantic_{config.semantic_weight}_geometric_{config.geometric_weight}_style_{config.style_weight}_dip_{config.use_dip}_straight_through_{config.straight_through}_by_distance_{config.by_distance}_prompt_{config.style_prompt}_{config.input_image}"
  filename = filename.replace(".", "_")
  filename = filename.replace(" ", "_")
  filename = filename.replace("/", "")
  print(f"Filename: {filename}")
  output_path_image = f"results/images/{filename}"
  output_path_graph = f"results/graphs/{filename}"

  # frame
  plt.figure(0)
  plt.imshow(frame_pil)
  plt.title(title)
  plt.savefig(output_path_image, bbox_inches='tight')
  plt.show()

  # loss graph
  iters = [x['iteration'] for x in checkpoints]
  plt.figure(1)
  plt.plot(iters, losses)
  plt.title(f"{config.input_image}\n{title}")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig(output_path_graph)
  plt.show()

  # training lapse
  frameSize = (600, 600)
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  output_path_lapse = f"results/lapses/{filename}.mp4"
  out = cv2.VideoWriter(str(output_path_lapse), fourcc, 2.0, frameSize)
  # font
  font = cv2.FONT_HERSHEY_SIMPLEX
  
  # org
  org = (35, 35)
  
  # fontScale
  fontScale = 1
  
  # Blue color in BGR
  color = (0, 0, 0)
  
  # Line thickness of 2 px
  thickness = 2
  for checkpoint in checkpoints:
    frame = torch.permute(checkpoint["frame"], (1, 2, 0)).numpy()
    img = to8b(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, frameSize, interpolation = cv2.INTER_NEAREST)
    iter = checkpoint["iteration"]
    loss = checkpoint["loss"]
    text = f"Iter: {iter:4}, Loss: {loss:6.4f}"
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    out.write(img)
  
  print("Finished rendering training lapse!")
  out.release()

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

"""# U-Net (From Voxel art Env):"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, small=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.small = small

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up2_small = Up(512, 64)
        self.up3 = Up(256, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        if self.small:
            x = self.up2_small(x, x3)
        else:
            x = self.up2(x, x3)
            x = self.up3(x, x2)
        logits = self.outc(x)
        return logits

"""### U-Net (skip model from DIP):"""
def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]
    
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

###############################
####### CANVAS CLASSES ########
###############################

class Canvas(nn.Module):
  def __init__(self, palette: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0, 
               straight_through=True):
    super().__init__()
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w
    weights = torch.empty((self.h, self.w, self.num_colors)).to(device)
    nn.init.xavier_normal_(weights)
    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    self.weight = torch.nn.parameter.Parameter(data=weights, requires_grad=True)
    self.palette = palette
    self.temperature = temperature
    self.st_softmax = ST_SoftMax(temperature)
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    if self.straight_through:
      norm_weights = self.st_softmax(self.weight)
    else:
      norm_weights = self.softmax(self.weight * self.temperature)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)
    return colors_upscaled

### Canvas Class by Distance: ###

class Canvas_by_Distance(nn.Module):
  def __init__(self, palette: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0, 
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
    self.st_softmax = ST_SoftMax(temperature)
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

class Canvas_DIP(nn.Module):
  def __init__(self, palette: Tensor,
               target: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0, 
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
    small = False
    if canvas_h == 32 and canvas_w == 32:
      small = True
    
    if image_init:
      self.backbone = UNet(3, self.num_colors, small=small).to(device)
      self.net_input = transforms.Resize((128, 128))(target.detach())
    else:
      self.backbone = UNet(self.num_colors, self.num_colors, small=small).to(device)
      self.net_input = get_noise(self.num_colors, 'noise', (128, 128)).to(device).detach()   

    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    if learn_colors:
      self.palette = torch.nn.parameter.Parameter(data=palette, requires_grad=True)
    else:
      self.palette = palette
    self.temperature = temperature
    self.st_softmax = ST_SoftMax(temperature)
    self.softmax = torch.nn.Softmax(dim=-1)
  
  def forward(self) -> Tensor:
    weight = self.backbone(self.net_input)
    weight = torch.permute(torch.squeeze(weight), (1, 2, 0))
    if self.straight_through:
      norm_weights = self.st_softmax(weight)
    else:
      norm_weights = self.softmax(weight * self.temperature)
    colors = torch.matmul(norm_weights, self.palette)
    colors = colors.permute(2, 0, 1)
    colors_upscaled = self.upsample(torch.unsqueeze(colors, 0))
    colors_upscaled = torch.squeeze(colors_upscaled)
    return colors_upscaled

### Canvas Class DIP by distance: ###

class Canvas_DIP_by_distance(nn.Module):
  def __init__(self, palette: Tensor,
               target: Tensor, 
               canvas_h: int,
               canvas_w: int,
               image_h: int,
               image_w: int,
               temperature: float=10000.0, 
               straight_through=True,
               image_init=True):
    super().__init__()
    self.num_colors = palette.size(dim=0)
    self.h = canvas_h
    self.w = canvas_w
    self.im_h = image_h
    self.im_w = image_w

    # Set UNet backbone
    small = False
    if canvas_h == 32 and canvas_w == 32:
      small = True
    
    if image_init:
      self.backbone = UNet(3, 3, small=small).to(device)
      self.net_input = transforms.Resize((128, 128))(target.detach())
    else:
      self.backbone = UNet(self.num_colors, 3, small=small).to(device)
      self.net_input = get_noise(self.num_colors, 'noise', (128, 128)).to(device).detach()   

    self.straight_through = straight_through
    self.upsample = torch.nn.Upsample(size=(self.im_h, self.im_w))
    self.palette = palette
    # Set repeated palette
    self.palette_repeated = torch.zeros((self.h, self.w, self.num_colors, 3)).to(device).detach()
    self.palette_repeated[:, :] = self.palette
    self.palette_repeated = torch.permute(self.palette_repeated, (-2, 0, 1, -1))
    self.temperature = temperature
    self.st_softmax = ST_SoftMax(temperature)
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

def canvas_selector(palette: Tensor,
                    target: Tensor,
                    use_dip: bool, 
                    straight_through: bool,
                    image_init: bool,
                    by_distance: bool, 
                    canvas_h: int, 
                    canvas_w: int, 
                    temperature: float,
                    ):
  _, _, im_h, im_w = target.size()
  if use_dip:
    if by_distance:
      print("Selecting: DIP by distance")
      canvas = Canvas_DIP_by_distance(palette,
                      target, 
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      straight_through,
                      image_init=image_init)
    else:
      print("Selecting: DIP")
      canvas = Canvas_DIP(palette,
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
      canvas = Canvas_by_Distance(palette,
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      straight_through)
    else:
      print("Selecting: No DIP")
      canvas = Canvas(palette,
                      canvas_h,
                      canvas_w,
                      im_h,
                      im_w,
                      temperature,
                      straight_through)
  
  return canvas

### Clip Visual Encoder class (from CLIPasso): ###

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output
        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]
        return featuremaps

############################
###### Loss Classes  #######
############################

class SemanticLoss(torch.nn.Module):
  def __init__(self, clip_model, target_features):
    super().__init__()
    self.clip_model = clip_model
    self.clip_model.eval()
    self.target_features = target_features
  
  def forward(self, x: Tensor) -> Tensor:
    # input here needs to be a clip encoding of the input image
    x_preprocessed = torch.unsqueeze(clip_tensor_preprocess(x), 0)
    input_features = self.clip_model.encode_image(x_preprocessed)
    return 1. - torch.cosine_similarity(input_features, self.target_features)

class L2Loss(torch.nn.Module):
  def __init__(self, target: Tensor):
    super().__init__()
    self.target = target
  
  def forward(self, x: Tensor) -> Tensor:
    return torch.mean((x - self.target)**2)

class GeometricLoss(torch.nn.Module):
  def __init__(self, clip_model, clip_layer_weights, target: Tensor):
    super().__init__()
    self.clip_model = clip_model
    self.clip_model.eval()
    self.clip_conv_layer_weights = clip_layer_weights
    self.visual_encoder = CLIPVisualEncoder(clip_model)
    self.target_conv_features = self.visual_encoder(target)
  
  def forward(self, x: Tensor) -> Tensor:
    # input here needs to be a clip encoding of the input image
    x_preprocessed = torch.unsqueeze(clip_tensor_preprocess(x), 0)
    input_conv_features = self.visual_encoder(x_preprocessed)
    conv_losses = [torch.square(x_conv - y_conv.detach()).mean() for x_conv, y_conv in
            zip(input_conv_features, self.target_conv_features)]
    loss = torch.tensor([0.0]).to(device)
    for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                loss = loss + conv_losses[layer] * w
    return loss

class PixelArtLoss(nn.Module):
  def __init__(self, 
               clip_model, 
               target: Tensor, 
               weight_dict: dict,
               conv_weights,
               style_prompt):
    super().__init__()
    self.clip_model = clip_model
    self.clip_model.eval()
    self.target_tensor = target
    self.target_preprocessed = clip_tensor_preprocess(target)
    self.target_features = clip_model.encode_image(self.target_preprocessed).detach()
    text = clip.tokenize(style_prompt).to(device)
    self.text_style_feature = clip_model.encode_text(text).detach()
    self.weight_table = weight_dict

    # set loss classes
    self.l2 = L2Loss(self.target_tensor)
    self.semantic_loss = SemanticLoss(clip_model, self.target_features)
    self.pa_style_loss = SemanticLoss(clip_model, self.text_style_feature)
    self.geometric_loss = GeometricLoss(clip_model, conv_weights, 
                                        self.target_preprocessed)

    # set losses to apply and weights
    self.losses_to_apply = []
    self.weights_to_apply = []

    if (weight_dict["l2"] != 0):
      self.losses_to_apply.append(self.l2)
      self.weights_to_apply.append(weight_dict["l2"])

    if (weight_dict["semantic"] != 0):
      self.losses_to_apply.append(self.semantic_loss)
      self.weights_to_apply.append(weight_dict["semantic"])
    
    if (weight_dict["style"] != 0):
      self.losses_to_apply.append(self.pa_style_loss)
      self.weights_to_apply.append(weight_dict["style"])
    
    if (weight_dict["geometric"] != 0):
      self.losses_to_apply.append(self.geometric_loss)
      self.weights_to_apply.append(weight_dict["geometric"])
    

  def forward(self, x: Tensor) -> Tensor:
    loss = torch.tensor([0.0]).to(device)
    #loss_input = torch.unsqueeze(clip_tensor_preprocess(x), 0)
    for loss_fn, w in zip(self.losses_to_apply, self.weights_to_apply):
      loss = loss + w * loss_fn(x)
    
    return loss

###########################
######    IMAGE IO    #####
########################### 

def get_target_and_palette(img_filepath: str, num_clusters: int):
  img = Image.open(img_filepath).convert("RGB")
  t_img = torch.unsqueeze(to_tensor(img), 0)
  open_cv_image = np.array(img) 
  
  # Convert RGB to BGR 
  open_cv_image = open_cv_image[:, :, ::-1].copy() 
  Z = np.float32(open_cv_image.reshape((-1,3)))

  # get centers
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = num_clusters
  _, _, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = center.reshape((1, -1, 3))
  center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
  center_pil = Image.fromarray(center.astype(np.uint8))
  t_centers = to_tensor(center_pil)
  t_centers = torch.squeeze(torch.permute(t_centers, (2, 1, 0)))

  return t_img.to(device), t_centers.to(device)

#################################
###### PLOTTING FUNCTIONS  ######
#################################

def plot_loss_graph(checkpoints: list, 
                    num_colors: int):
  plt_filename = f"loss_{num_colors}.png"
  output_path = f"/content/drive/MyDrive/Personal Thesis stuff/PixelArtClipasso2/{plt_filename}"
  losses = [x['loss'] for x in checkpoints]
  iters = [x['iteration'] for x in checkpoints]
  plt.plot(iters, losses)
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig(output_path)
  plt.show()


if __name__ == "__main__":
    main()