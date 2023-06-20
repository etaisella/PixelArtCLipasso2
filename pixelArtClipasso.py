import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from datetime import datetime
from PIL import Image
import click
import clip
from canvas import *
from loss import PixelArtLoss
from easydict import EasyDict
import wandb



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
@click.option("--old_method", type=click.BOOL, required=False, default=False,
              help="determines whether to use the old method of argmax, which uses temperature",
              show_default=False)
@click.option("--no_palette_mode", type=click.BOOL, required=False, default=False,
              help="determines whether to not use a palette and just directly learn a canvas",
              show_default=False)

# Losses
@click.option("--l2_weight", type=click.FLOAT, required=False, default=1.0,
              help="l2 weight", show_default=True)
@click.option("--style_weight", type=click.FLOAT, required=False, default=0.0,
              help="style weight", show_default=True)
@click.option("--style_prompt", type=click.STRING, required=False, default="none",
              help="style input prompt", show_default=True)
@click.option("--semantic_weight", type=click.FLOAT, required=False, default=0.0,
              help="semantic weight", show_default=True)
@click.option("--geometric_weight", type=click.FLOAT, required=False, default=0.0,
              help="geometric weight", show_default=True)
@click.option("--shift_aware_weight", type=click.FLOAT, required=False, default=0.0,
              help="shift aware weight", show_default=True)

# Training
@click.option("--lr", type=click.FLOAT, required=False, default=0.00005,
              help="learning rate", show_default=True)
@click.option("--start_anneal_iter", type=click.INT, required=False, default=1500,
              help="Iteration in which we start to anneal learning rate", show_default=True)
@click.option("--lr_gamma", type=click.FLOAT, required=False, default=1.0,
              help="gamma by which we reduce learning rate", show_default=True)
@click.option("--save_freq", type=click.INT, required=False, default=100,
              help="frequency to save results in", show_default=True)
@click.option("--epochs", type=click.INT, required=False, default=16000,
              help="number of epochs", show_default=True)
@click.option("--start_semantics_iter", type=click.INT, required=False, default=1,
              help="the epoch where we start using semantic losses", show_default=True)


# fmt: on
# -------------------------------------------------------------------------------------

def main(**kwargs) -> None:
    #task = Task.init(project_name='PixelArtClipasso v2', task_name='Experiment 0')

    # load the requested configuration for the training
    config = EasyDict(kwargs)
    wandb.init(project='pixelArtClipasso v2', entity="etaisella",
                   config=dict(config), name="test " + str(datetime.now()), 
                   id=wandb.util.generate_id())

    # freeze clip parameters
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    # get image and palette
    image_path = "inputs" / Path(config.input_image)
    target, palette = get_target_and_palette(image_path, config.num_colors)
    wandb.log({"input": wandb.Image(target)}, step=0)
    wandb.log({"palette": wandb.Image(torch.unsqueeze(torch.permute(palette, (1, 0)), dim=-2))}, step=0)

    # get canvas class
    canvas = canvas_selector(device, palette, target, config.use_dip, config.straight_through,
                            config.init_image, config.by_distance, config.canvas_h, config.canvas_w, 
                            config.temperature, config.old_method, config.no_palette_mode)
    
    # set losses
    loss_dict = dict.fromkeys(["l2", "semantic", "style", "geometric", "shift_aware"], 
                          torch.tensor([0.0]).to(device))
    loss_dict_l2_only = dict.fromkeys(["l2", "semantic", "style", "geometric", "shift_aware"], 
                          torch.tensor([0.0]).to(device))
    loss_dict["semantic"] = torch.tensor([config.semantic_weight]).to(device)
    loss_dict["style"] = torch.tensor([config.style_weight]).to(device)
    loss_dict["geometric"] = torch.tensor([config.geometric_weight]).to(device)
    loss_dict['l2'] = torch.tensor([config.l2_weight]).to(device)
    loss_dict['shift_aware'] = torch.tensor([config.shift_aware_weight]).to(device)
    clip_conv_layer_weights = [0, 0.0, 1.0, 1.0, 0]

    print(f"Loss dict: \n {loss_dict}")

    # loss function with semantics as well as l2
    loss_fn_full = PixelArtLoss(clip_model,
                           device,
                           target,
                           loss_dict,
                           clip_conv_layer_weights,
                           config.style_prompt,
                           config.canvas_h, 
                           config.canvas_w,)
    
    # loss function with l2 only
    loss_dict_l2_only['l2'] = torch.tensor([1.0]).to(device)
    loss_fn_l2_only = PixelArtLoss(clip_model,
                           device,
                           target,
                           loss_dict_l2_only,
                           clip_conv_layer_weights,
                           config.style_prompt,
                           config.canvas_h, 
                           config.canvas_w,)

    # set optimizer & scheduler
    optimizer = torch.optim.Adam(canvas.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)

    #----------------#
    # Training Loop:
    #----------------#

    checkpoints = []
    for iter in range(config.epochs):
      optimizer.zero_grad()
      output, output_pa = canvas()
      if (iter >= config.start_semantics_iter):
        loss, loss_dict = loss_fn_full(output)
      else:
        loss, loss_dict = loss_fn_l2_only(output)
      loss.backward()
      optimizer.step()
      if (iter % config.save_freq == 0) or (iter == config.epochs - 1):
        checkpoint = {}
        checkpoint["iteration"] = iter
        checkpoint["loss"] = loss.item()
        checkpoint["frame"] = torch.squeeze(output).cpu().detach()
        checkpoint["pixel art frame"] = torch.squeeze(output_pa).cpu().detach()
        checkpoints.append(checkpoint)
        print(f"Iter: {iter}, Loss: {loss.item()}")

        # LR Scheduling:
        if iter >= config.start_anneal_iter:
          scheduler.step()

        # wandb logging:
        wandb.log({"current learning rate:": float(scheduler.get_last_lr()[0])}, step=iter)
        wandb.log({"overall loss": loss.item()}, step=iter)
        wandb.log({"Output": wandb.Image(checkpoint["frame"])}, step=iter)
        wandb.log({"Output PA": wandb.Image(checkpoint["pixel art frame"])}, step=iter)
        for k in loss_dict.keys():
          wandb.log({k + "_loss": loss_dict[k]}, step=iter)
        
    # Save Results:
    plot_results(checkpoints, config)
    wandb.finish()

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
  title = f"{config.num_colors} colors \n l2 {config.l2_weight}, semantic {config.semantic_weight},geometric {config.geometric_weight}, style {config.style_weight}\n use dip {config.use_dip},straight through {config.straight_through} \nby distance {config.by_distance}, image init {config.init_image} \nstyle prompt - {config.style_prompt}, old method {config.old_method}"
  filename = f"{config.num_colors}_colors_l2_{config.l2_weight}_semantic_{config.semantic_weight}_geometric_{config.geometric_weight}_style_{config.style_weight}_dip_{config.use_dip}_straight_through_{config.straight_through}_by_distance_{config.by_distance}_image_init_{config.init_image}_prompt_{config.style_prompt}_oldm_{config.old_method}_{config.input_image}"
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