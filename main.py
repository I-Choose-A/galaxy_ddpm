import os
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from unet import UNet
from scheduler import GradualWarmupScheduler
from dataset import SDSS


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # Dataset setup
    astronomical_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.nan_to_num(x, nan=0.0)),
        transforms.Lambda(lambda x: np.clip(x, -10, 1000)),
        transforms.Lambda(lambda x: np.transpose(x, (2, 0, 1))),
        transforms.Lambda(lambda x: np.log1p(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.00615956, 0.02047303, 0.03759114, 0.05205064, 0.05791357],
                             std=[0.04185153, 0.07266889, 0.1180148, 0.15163979, 0.21814607])
    ])

    dataset = SDSS(transform=astronomical_transform)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)

    # Model initialization
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"],
                     ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"],
                     dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(
            os.path.join(modelConfig["save_weight_dir"],
                         modelConfig["training_load_weight"]),
            map_location=device))

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"],
        modelConfig["T"]).to(device)

    # Training loop
    print(f"Training started for {modelConfig['epoch']} epochs")
    for epoch in range(modelConfig["epoch"]):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0).sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # Print batch progress every 10% of dataset
            if batch_idx % max(1, len(dataloader) // 10) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{modelConfig['epoch']} | "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Batch Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")

        warmUpScheduler.step()

        # Epoch summary
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1} completed | "
              f"Avg Loss: {avg_epoch_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        torch.save(net_model.state_dict(),
                   os.path.join(modelConfig["save_weight_dir"],
                                f'ckpt_{epoch}_.pt'))

    print("Training completed successfully")


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])


if __name__ == '__main__':
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 2,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
    }

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
