# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained EnDora.
"""
import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from diffusers.models import AutoencoderKL
import imageio
from omegaconf import OmegaConf

import numpy as np
import random

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import video_transforms
from torchvision import transforms
from PIL import Image

import glob
import models.vision_transformer as vits

def load_model(device, pretrained_path):
    model = vits.__dict__["vit_small"](
        patch_size=8, num_classes=0
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    state_dict = torch.load(pretrained_path, map_location="cpu")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_path, msg
        )
    )

    return model

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    transform_col = transforms.Compose([
        video_transforms.ToTensorVideo(),  # TCHW
        video_transforms.RandomHorizontalFlipVideo(),
        video_transforms.UCFCenterCropVideo(args.image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    image_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.ckpt is None:
        print('ckpt Path Not available')
        exit(-1)

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    print(args.model)
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom EnDora checkpoint from train.py:
    '''
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    '''
    states = torch.load(args.ckpt)
    model.load_state_dict(states['ema'])

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)

    # Labels to condition the model with (feel free to change):

    # Create sampling noise:
    for idx in range (1, 17):
        if args.use_fp16:
            z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
        else:
            z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

        vframes_m, aframes_m, info_m = torchvision.io.read_video(
            filename='/home/work/polypgen/CVC-ClinicDB/mask_video/' + str(idx).zfill(5) + '.mp4',
            pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes_m)

        temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval)
        start_frame_ind, end_frame_ind = temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= args.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int)

        # Sampling mask video frames
        video_m = vframes_m[frame_indice]
        # videotransformer data proprecess
        video_m = transform_col(video_m)  # T C H W

        c = video_m.contiguous().cuda()

        vae.requires_grad_(False)

        # c = rearrange(c, 'b f c h w -> (b f) c h w').contiguous()
        c = vae.encode(c).latent_dist.sample().mul_(0.18215)
        c = rearrange(c, '(b f) c h w -> b f c h w', b=1).contiguous()

        # Setup classifier-free guidance:
        # z = torch.cat([z, c], 0)
        if using_cfg:
            print('using cfg')
            z = torch.cat([z, c], 0)
            y = torch.randint(0, args.num_classes, (1,), device=device)
            y_null = torch.tensor([101] * 1, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            sample_fn = model.forward_with_cfg
        else:
            sample_fn = model.forward
            model_kwargs = dict(y=None, use_fp16=args.use_fp16)

        model_kwargs["y_image"] = c

        # Sample images:
        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

        print(samples.shape)
        if args.use_fp16:
            samples = samples.to(dtype=torch.float16)
        b, f, c, h, w = samples.shape
        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
        samples = vae.decode(samples / 0.18215).sample
        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
        # Save and display images:

        if not os.path.exists(args.save_video_path):
            os.makedirs(args.save_video_path)

        print(samples.shape, "!!!")
        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, 'sample_' + str(idx).zfill(2) + '.mp4')
        print(video_save_path, video_.shape)
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="/home/work/polypgen/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
