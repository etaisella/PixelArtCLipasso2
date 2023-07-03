from transformers import CLIPTextModel, CLIPTokenizer, logging, AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, \
    StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import numpy as np
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from torch import Tensor
from jaxtyping import Float, Int

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusionControl(nn.Module):
    def __init__(self, device, control_image_path,
                 t_sched_start = 1500,
                 t_sched_freq = 500,
                 t_sched_gamma = 1.0,
                 output_path = None):
        super().__init__()
        self.device = device
        self.weights_dtype = torch.float16
        self.diffusion_steps = 20
        self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

        # set up timestep scheduler stuff
        self.t_sched_start = t_sched_start
        self.t_sched_freq = t_sched_freq
        self.t_sched_gamma = t_sched_gamma

        # set up control image
        image = load_image(str(control_image_path))
        image = torch.from_numpy(np.array(image)).to(self.device).unsqueeze(0)
        self.control_image = self.prepare_image_cond(image, output_path / f"control_image.png")

        #image = load_image(str(control_image_path))
        #image = self.hed(image)

        ## save control image
        #if output_path is not None:
        #    ctrl_img_path = output_path / f"control_image.png"
        #    print(f"Saving control image to - {ctrl_img_path}")
        #    image.save(ctrl_img_path)

        ## convert image from PIL to tensor
        #image = torch.from_numpy(np.array(image)).to(self.device)
        

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.sd_name = "runwayml/stable-diffusion-v1-5"

        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=None,
        )
        self.scheduler.set_timesteps(self.diffusion_steps)

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        #pipe.enable_xformers_memory_efficient_attention()
        
        self.pipe.enable_model_cpu_offload()
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        #image = pipe("an oil painting of a cute baby girl", image, num_inference_steps=20).images[0]

        ## Create model
        #self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", use_auth_token=use_auth_token).to(
        #    self.device)
        #self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",
        #                                               use_auth_token=use_auth_token)
        #self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
        #                                                  use_auth_token=use_auth_token).to(self.device)
        #self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
        #                                                 use_auth_token=use_auth_token).to(
        #    self.device)

        #self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",
        #                                               use_auth_token=use_auth_token)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, self.device)

        self.min_step_ratio = 0.02
        self.min_step = int(self.num_train_timesteps * self.min_step_ratio)

        self.max_step_ratio = 0.98
        self.max_step = int(self.num_train_timesteps * self.max_step_ratio)

        self.guidance_scale = 7.5
        self.condition_scale = 1.5

        print(f'[INFO] loaded controlnet!')

    def get_max_step_ratio(self):
        return self.max_step_ratio

    def prepare_image_cond(self, cond_rgb, output_path):
        cond_rgb = (
            (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
        )
        detected_map = self.hed(cond_rgb)

        # save detected map
        if output_path is not None:
            detected_map.save(output_path)

        control = (
            torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
        )
        control = control.unsqueeze(0)
        control = control.permute(0, 3, 1, 2)
        
        return F.interpolate(control, (512, 512), mode="bilinear", align_corners=False)

    def get_text_embeds(self, prompt, negative_prompt = ''):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sd_name, subfolder="tokenizer"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sd_name, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prompt, negative_prompt: [str]
        #prompt_embeds = self.pipe._encode_prompt(
        #    prompt,
        #    device=self.device,
        #    num_images_per_prompt=1,
        #    do_classifier_free_guidance=True,
        #    negative_prompt=negative_prompt,
        #    prompt_embeds=None,
        #    negative_prompt_embeds=None,
        #)

        #self.prompt = prompt

        ## Tokenize text and get embeddings
        #text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        #with torch.no_grad():
        #    text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        ## Do the same for unconditional embeddings
        #uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        #with torch.no_grad():
        #    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        ## Cat for final embeddings
        #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    def compute_grad_sds(
        self,
        text_embeddings,
        latents,
        image_cond,
        t,
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond,
                condition_scale=self.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=7.5, global_step=-1, logvar=None):
        # TODO: check if guidance_scale should be 100 or 7.5
        # schedule max step:
        if global_step >= self.t_sched_start and global_step % self.t_sched_freq == 0:
            self.max_step_ratio = self.max_step_ratio * self.t_sched_gamma

            # if self.max_step_ratio < self.min_step_ratio * 2:

            if self.max_step_ratio < 0.27:
                #self.max_step_ratio = self.min_step_ratio * 2 # don't let it get too low!
                self.max_step_ratio = 0.27 # don't let it get too low!
            else:
                print(f"Updating max step to {self.max_step_ratio}")

        self.max_step = int(self.num_train_timesteps * self.max_step_ratio)

        # 0. Default height and width to unet
        height = 512
        width = 512

        # 1. Check inputs. Raise error if not correct
        #self.pipe.check_inputs(
        #    prompt=self.prompt,
        #    image=self.control_image,
        #    height=height,
        #    width=width,
        #    callback_steps=1,
        #    negative_prompt=None,
        #    prompt_embeds=None,
        #    negative_prompt_embeds=None,
        #    controlnet_conditioning_scale=1.0,
        #)

        # 2. Prepare control image
        #image = self.pipe.prepare_image(
        #    image=self.control_image,
        #    width=width,
        #    height=height,
        #    batch_size=1,
        #    num_images_per_prompt=1,
        #    device=self.device,
        #    dtype=self.controlnet.dtype,
        #    do_classifier_free_guidance=True,
        #)
        
        # 3. Prepare timesteps
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # 4. Encode image
        pred_rgb_hw = F.interpolate(pred_rgb, (height, width), mode='nearest')
        latents = self.encode_imgs(pred_rgb_hw)

        # 5. Predict grad
        grad = self.compute_grad_sds(text_embeddings, latents, self.control_image, t)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / 1
        
        #with torch.no_grad():
#
        #    # add noise
        #    noise = torch.randn_like(latents)
        #    latents_noisy = self.scheduler.add_noise(latents, noise, t)
        #    # pred noise
        #    latent_model_input = torch.cat([latents_noisy] * 2)
        #    
        #    # controlnet(s) inference
        #    down_block_res_samples, mid_block_res_sample = self.controlnet(
        #        latent_model_input,
        #        t,
        #        encoder_hidden_states=text_embeddings,
        #        controlnet_cond=self.control_image,
        #        conditioning_scale=self.condition_scale,
        #        return_dict=False,
        #    )
#
        #    # predict the noise residual
        #    noise_pred = self.unet(
        #        latent_model_input,
        #        t,
        #        encoder_hidden_states=text_embeddings,
        #        cross_attention_kwargs=None,
        #        down_block_additional_residuals=down_block_res_samples,
        #        mid_block_additional_residual=mid_block_res_sample,
        #    ).sample


        #self.scheduler.set_timesteps(num_inference_steps, device=device)
        #timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        #num_channels_latents = self.unet.config.in_channels
        #latents = self.prepare_latents(
        #    1,
        #    num_channels_latents,
        #    height,
        #    width,
        #    text_embeddings.dtype,
        #    device=self.device,
        #    generator,
        #    latents,
        #)

        #pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='nearest')


        ## timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        #t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        ## encode image into latents with vae, requires grad!
        ## _t = time.time()
        #latents = self.encode_imgs(pred_rgb_512)
        ## torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        ## predict the noise residual with unet, NO grad!
        ## _t = time.time()
        #with torch.no_grad():
        #    # add noise
        #    noise = torch.randn_like(latents)
        #    latents_noisy = self.scheduler.add_noise(latents, noise, t)
        #    # pred noise
        #    latent_model_input = torch.cat([latents_noisy] * 2)
        #    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        ## torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
#
        ## w(t), sigma_t^2
        #w = (1 - self.alphas[t])
        ## w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        #grad = w * (noise_pred - noise)
#
        ## clip grad for stable training?
        ## grad = grad.clamp(-10, 10)
        #grad = torch.nan_to_num(grad)
#
        #if logvar != None:
        #    grad = grad * torch.exp(-1 * logvar)
#
        ## since we omitted an item in grad, we need to use the custom function to specify the gradient
        ## _t = time.time()
        #loss = SpecifyGradient.apply(latents, grad)
        ## torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss_sds

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.cuda.amp.autocast(enabled=False)
    def encode_imgs(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs