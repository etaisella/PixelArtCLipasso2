import torch
import torch.nn as nn
import collections
import clip
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
               style_prompt):
    super().__init__()
    self.device = device
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
    self.geometric_loss = GeometricLoss(clip_model, device, conv_weights, 
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
    loss = torch.tensor([0.0]).to(self.device)
    #loss_input = torch.unsqueeze(clip_tensor_preprocess(x), 0)
    for loss_fn, w in zip(self.losses_to_apply, self.weights_to_apply):
      loss = loss + w * loss_fn(x)
    
    return loss