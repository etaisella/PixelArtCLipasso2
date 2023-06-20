import torch
import torch.nn as nn
import collections
import clip
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image
from torchvision import transforms
from torch import Tensor

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# clip preprocess for tensor
def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

clip_tensor_preprocess = clip_transform(224)

def to8b(x: np.array) -> np.array:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

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

class ShiftAwareLoss(torch.nn.Module):
  def __init__(self, target: Tensor, canvas_h: int, canvas_w: int):
    super().__init__()
    self.step = 0
    n, c, h, w = target.shape
    self.block_size_sq = int(h / canvas_h)
    self.target = target
    target_nhwc = torch.permute(target, (0, 2, 3, 1))
    self.background_mask = torch.squeeze(torch.logical_and(torch.logical_and(target_nhwc[..., 0] == 1.0, \
      target_nhwc[..., 1] == 1.0), target_nhwc[..., 2] == 1.0))
    self.loss_class = torch.nn.L1Loss(reduction='none')
    self.upsample = torch.nn.Upsample(size=(h, w))
  
  def forward(self, x: Tensor) -> Tensor:
    diff_img = torch.squeeze(torch.mean(self.loss_class(x, torch.squeeze(self.target)), dim=0))
    diff_img_orig = torch.clone(diff_img)
    diff_img[self.background_mask] = 100.0
    max_pool = nn.MaxPool2d(self.block_size_sq, stride=self.block_size_sq)
    min_image_small = -max_pool(-torch.unsqueeze(diff_img, dim=0))
    min_image = torch.squeeze(self.upsample(torch.unsqueeze(min_image_small, dim=0)))
    big_background_pixel_mask = (min_image == 100.0)
    min_image[big_background_pixel_mask] = diff_img_orig[big_background_pixel_mask]

    if self.step % 50 == 0:
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
      diff_image_np = to8b(diff_img_orig.detach().cpu().numpy())
      ax1.imshow(diff_image_np, cmap='jet')
      ax1.set_title("Diff Image")

      min_image_np = to8b(min_image.detach().cpu().numpy())
      ax2.imshow(min_image_np, cmap='jet')
      ax2.set_title("Min diff Image")

      plt.tight_layout()
      wandb.log({"SA image": wandb.Image(plt)}, step=self.step)
      plt.savefig('my_plot.png')
      plt.close()


    self.step = self.step + 1
    return torch.mean(min_image)

class SemanticStyleLoss(torch.nn.Module):
  def __init__(self, clip_model, device):
    super().__init__()
    self.device = device
    self.loss_fn = torch.nn.BCELoss()
    self.clip_model = clip_model
    text_prompts = ["a vibrant pink pixel art flamingo with black legs", 
                    "pixelized image of a flamingo",
                    "downsampled image of a flamingo"]
    text_prompts = [x.strip() for x in text_prompts]

    with torch.no_grad():
      self.target_features = clip.tokenize(text_prompts).to(device)
  
  def forward(self, x: Tensor) -> Tensor:
    # input here needs to be a clip encoding of the input image
    x_preprocessed = torch.unsqueeze(clip_tensor_preprocess(x), 0)
    logits_per_image, _ = self.clip_model(x_preprocessed, self.target_features)
    probs = logits_per_image.softmax(dim=-1).float()
    targets = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).float()
    loss = self.loss_fn(probs, targets)
    return loss

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
  def __init__(self, clip_model, device, clip_layer_weights, target: Tensor):
    super().__init__()
    self.device = device
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
    loss = torch.tensor([0.0]).to(self.device)
    for layer, w in enumerate(self.clip_conv_layer_weights):
            if w:
                loss = loss + conv_losses[layer] * w
    return loss

class PixelArtLoss(nn.Module):
  def __init__(self,
               clip_model,
               device,
               target: Tensor, 
               weight_dict: dict,
               conv_weights,
               style_prompt,
               canvas_h,
               canvas_w):
    super().__init__()
    self.device = device
    self.clip_model = clip_model
    self.clip_model.eval()
    self.target_tensor = target
    self.target_preprocessed = clip_tensor_preprocess(target)
    if style_prompt == "none":
      self.target_features = clip_model.encode_image(self.target_preprocessed).detach()
    else:
      text = clip.tokenize(style_prompt).to(device)
      self.target_features = clip_model.encode_text(text).detach()
    self.text_style_feature = clip_model.encode_text(text).detach()
    self.weight_table = weight_dict

    # set loss classes
    self.l2 = L2Loss(self.target_tensor)
    self.semantic_loss = SemanticLoss(clip_model, self.target_features)
    #self.pa_style_loss = SemanticLoss(clip_model, self.text_style_feature)
    self.pa_style_loss = SemanticStyleLoss(clip_model, device)
    self.geometric_loss = GeometricLoss(clip_model, device, conv_weights, 
                                        self.target_preprocessed)
    self.shift_aware_loss = ShiftAwareLoss(target, canvas_h, canvas_w)

    # set losses to apply and weights
    self.losses_to_apply = []
    self.weights_to_apply = []
    self.loss_names = []

    if (weight_dict["l2"] != 0):
      self.loss_names.append("l2")
      self.losses_to_apply.append(self.l2)
      self.weights_to_apply.append(weight_dict["l2"])

    if (weight_dict["semantic"] != 0):
      self.loss_names.append("semantic")
      self.losses_to_apply.append(self.semantic_loss)
      self.weights_to_apply.append(weight_dict["semantic"])
    
    if (weight_dict["style"] != 0):
      self.loss_names.append("style")
      self.losses_to_apply.append(self.pa_style_loss)
      self.weights_to_apply.append(weight_dict["style"])
    
    if (weight_dict["geometric"] != 0):
      self.loss_names.append("geometric")
      self.losses_to_apply.append(self.geometric_loss)
      self.weights_to_apply.append(weight_dict["geometric"])
    
    if (weight_dict["shift_aware"] != 0):
      self.loss_names.append("shift_aware")
      self.losses_to_apply.append(self.shift_aware_loss)
      self.weights_to_apply.append(weight_dict["shift_aware"])
    

  def forward(self, x: Tensor, x_pa: Tensor=None) -> Tensor:
    loss_dict = {}
    loss = torch.tensor([0.0]).to(self.device)
    for loss_fn, w, name in zip(self.losses_to_apply, self.weights_to_apply, self.loss_names):
      if name == "shift_aware" and x_pa != None:
        curr_loss = loss_fn(x_pa)
      else:
        curr_loss = loss_fn(x)
      loss_dict[name] = w * curr_loss.item()
      loss = loss + w * curr_loss
    
    return loss, loss_dict