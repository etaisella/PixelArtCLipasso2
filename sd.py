from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, \
    StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from sd_control import StableDiffusionControl
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd

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


class StableDiffusion(nn.Module):
    def __init__(self, device,
                 sd_version='2.1',
                 hf_key=None,
                 t_sched_start = 1500,
                 t_sched_freq = 500,
                 t_sched_gamma = 1.0, auth_token=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.t_sched_start = t_sched_start
        self.t_sched_freq = t_sched_freq
        self.t_sched_gamma = t_sched_gamma

        print(f'[INFO] loading stable diffusion...')

        use_auth_token = False
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
            use_auth_token = auth_token
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", use_auth_token=use_auth_token).to(
            self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",
                                                       use_auth_token=use_auth_token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",
                                                          use_auth_token=use_auth_token).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",
                                                         use_auth_token=use_auth_token).to(
            self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",
                                                       use_auth_token=use_auth_token)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, self.device)

        self.min_step_ratio = 0.02
        self.min_step = int(self.num_train_timesteps * self.min_step_ratio)

        self.max_step_ratio = 0.98
        self.max_step = int(self.num_train_timesteps * self.max_step_ratio)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_max_step_ratio(self):
        return self.max_step_ratio

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

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
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, global_step=-1, logvar=None):
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

        # interp to 512x512 to be fed into vae.
        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='nearest')
        #pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        if logvar != None:
            grad = grad * torch.exp(-1 * logvar)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # _t = time.time()
        loss = SpecifyGradient.apply(latents, grad)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss

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

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

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


class scoreDistillationLoss(nn.Module):
    def __init__(self,
                 device,
                 prompt,
                 image_height,
                 image_width,
                 t_sched_start = 1500,
                 t_sched_freq = 500,
                 t_sched_gamma = 1.0,
                 directional = False,
                 sds_control = False,
                 control_image_path = None,
                 output_path = None,):
        super().__init__()
        self.dir_to_indx_dict = {}
        self.directional = directional
        self.image_height = image_height
        self.image_width = image_width

        # get sd model
        if sds_control:
            print("Using SDS Control")
            self.sd_model = StableDiffusionControl(device,
                                                   control_image_path=control_image_path,
                                                   t_sched_start=t_sched_start,
                                                   t_sched_freq=t_sched_freq,
                                                   t_sched_gamma=t_sched_gamma,
                                                   output_path=output_path)
        else:
            print("Using SD SDS")
            self.sd_model = StableDiffusion(device,
                                            "2.0",
                                            t_sched_start=t_sched_start,
                                            t_sched_freq=t_sched_freq,
                                            t_sched_gamma=t_sched_gamma)

        # encode text
        if directional:
            self.text_encodings = {}
            for dir_prompt in ['side', 'overhead', 'back', 'front']:
                print(f"Encoding text for \'{dir_prompt}\' direction")
                modified_prompt = prompt + f", {dir_prompt} view"
                self.text_encodings[dir_prompt] = self.sd_model.get_text_embeds(modified_prompt, '')
        else:
            self.text_encoding = self.sd_model.get_text_embeds(prompt, '')

    def get_current_max_step_ratio(self):
        return self.sd_model.get_max_step_ratio()

    def forward(self, output, directions=None, global_step=-1, logvars=None):
        loss = 0
        if self.directional:
            assert (directions != None), f"Must supply direction if SDS loss is set to directional mode"
        # format output images
        #print(f"output shape - {output.shape}")
        #out_imgs = torch.reshape(output, (-1, self.image_height, self.image_width, 3))
        #out_imgs = out_imgs.permute((0, 3, 1, 2))
        #print(f"out_imgs - {out_imgs.shape}")
        out_imgs = output.unsqueeze(dim=0)
        
        # perform training step
        if not self.directional:
            loss = self.sd_model.train_step(self.text_encoding, out_imgs, global_step=global_step, logvar=logvars)
        else:
            for idx, dir_prompt in enumerate(directions):
                if logvars != None:
                    logvar = logvars[idx]
                else:
                    logvar = None
                encoding = self.text_encodings[dir_prompt]
                loss = loss + self.sd_model.train_step(encoding, out_imgs, global_step=global_step, logvar=logvar)

        return loss