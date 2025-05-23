import torch

# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from transformers import CLIPProcessor, CLIPModel
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema,
                   requires_grad, cleanup, create_tensorboard,
                   write_tensorboard, setup_distributed, get_experiment_dir)
import models.vision_transformer as vits


#################################################################################
#                                  Training Loop                                #
#################################################################################

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


def main(args, port, pretrained_weights, mode, prr_weight):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    os.environ['MASTER_PORT'] = str(port)
    # Setup DDP:
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = rank
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = 1
        model_string_name = args.model.replace("/", "-")  # e.g., EnDora-XL/2 --> EnDora-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    # Note that parameter initialization is done within the EnDora constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    dino = load_model(device=device, pretrained_path=pretrained_weights)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if args.extras == 78:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
        s_vae_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        s_vae_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(
            len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    if args.fixed_spatial:
        trainable_modules = (
            "attn_temp",
        )
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if name.endswith(tuple(trainable_modules)):
                for params in module.parameters():
                    logger.info("WARNING: Only train {} parametes!".format(name))
                    params.requires_grad = True
        logger.info("WARNING: Only train {} parametes!".format(trainable_modules))

    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    s_vae_m.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # Get the most recent checkpoint
        logger.info(f"Resuming from checkpoint")
        states = torch.load(args.resume_from_checkpoint)
        model.module.load_state_dict(states['ema'])
        del states
        train_steps = 20000

        first_epoch = train_steps // num_update_steps_per_epoch

    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):

            x = video_data['video'].to(device, non_blocking=True)
            if args.extras == 3:
                c = video_data['video_mask'].to(device, non_blocking=True)

            img = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
            patch_size = 8
            # modified by piang
            w, h = (
                img.shape[-2] - img.shape[-2] % patch_size,
                img.shape[-1] - img.shape[-1] % patch_size,
            )
            img = img[:, :, :w, :h]

            special_list = [2, 5, 8, 11]
            attentions = dino.get_special_layers(img.to(device), special_list)
            attentions = [item[:, 1:, :] for item in attentions]

            video_name = video_data['video_name']
            with torch.no_grad():

                # concatenate input with mask
                # x = torch.concatenate([x, c], dim=1)

                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

                if args.extras == 3:
                    c = rearrange(c, 'b f c h w -> (b f) c h w').contiguous()
                    c = vae.encode(c).latent_dist.sample().mul_(0.18215)
                    c = rearrange(c, '(b f) c h w -> b f c h w', b=b).contiguous()

            if args.extras == 2:
                model_kwargs = dict(y=video_name)  # tav unet
            else:
                model_kwargs = dict(y=None, use_image_num=args.use_image_num)

            model_kwargs["attentions"] = attentions
            model_kwargs["special_list"] = special_list
            model_kwargs["mode"] = mode
            if args.extras == 3:
                model_kwargs["y_image"] = c

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_mse = loss_dict["loss"].mean()
            loss_prr = loss_dict["prr"]
            if loss_prr != -1:
                loss = loss_mse + prr_weight * loss_prr
            else:
                loss = loss_mse
            loss.backward()

            if train_steps < args.start_clip_iter:  # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Total L: {avg_loss:.4f}, MSE L: {loss_dict['mse'].mean():.4f}, VB L: {loss_dict['vb'].mean():.4f}, PRR L: {loss_dict['prr'].data:.4f},"
                    f" Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save EnDora checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "ema": ema.state_dict()
                    }

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train EnDora-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--port", type=int, default=6035)
    parser.add_argument('--pretrained_weights', type=str, default="/path/to/pretrained/dino-model")
    parser.add_argument("--mode", type=str, default="type_cnn", choices=["type0", "type1", "type2", "type_cnn"])
    parser.add_argument("--prr_weight", type=float, default=0.1)
    args = parser.parse_args()
    main(OmegaConf.load(args.config), args.port, args.pretrained_weights, args.mode, args.prr_weight)
