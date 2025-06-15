import os
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
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
        transforms.Lambda(lambda x: np.log1p(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.00615956, 0.02047303, 0.03759114, 0.05205064, 0.05791357],
                             std=[0.04185153, 0.07266889, 0.1180148, 0.15163979, 0.21814607])
    ])

    dataset = SDSS(
        images_path=modelConfig["images_path"],
        conditions_path=modelConfig["conditions_path"],
        channels=modelConfig["selected_channel"],
        transform=astronomical_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

    # Model initialization
    net_model = UNet(
        T=modelConfig["T"],
        img_ch=modelConfig["num_img_channel"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"]),
                       map_location=device))

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0,
        last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )

    trainer = GaussianDiffusionTrainer(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"]
    ).to(device)

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

        # Save checkpoint, only save last checkpoint
        if epoch == modelConfig["epoch"] - 1:
            torch.save(net_model.state_dict(),
                       os.path.join(modelConfig["save_weight_dir"],
                                    f'ckpt_{epoch}_.pt'))

    print("Training completed successfully")


def inverse_astronomical_transform(tensor, channels=None):
    """Inverse transformation: restoring the original astronomical data from the normalized tensor"""
    original_channels = ["u", "g", "r", "i", "z"]
    if channels:
        channels = [original_channels.index(ch) for ch in channels]
    else:
        channels = [0, 1, 2, 3, 4]
    sdss_mean = [0.00615956, 0.02047303, 0.03759114, 0.05205064, 0.05791357]
    sdss_std = [0.04185153, 0.07266889, 0.1180148, 0.15163979, 0.21814607]

    # inverse normalization
    mean = torch.tensor([sdss_mean[i] for i in channels], device=tensor.device)
    std = torch.tensor([sdss_std[i] for i in channels], device=tensor.device)
    denormalized = tensor * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    # inverse Log1p
    denormalized = torch.expm1(denormalized)  # exp(x) - 1

    return denormalized


def sampling(modelConfig: Dict):
    # load model and sampling
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(
            T=modelConfig["T"],
            img_ch=modelConfig["num_img_channel"],
            ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=0.
        )
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]),
                          map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model,
            modelConfig["beta_1"],
            modelConfig["beta_T"],
            modelConfig["T"]
        ).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], modelConfig["num_img_channel"], 64, 64],
            device=device
        )

        sampledImgs = sampler(noisyImage)

        # Inverse transform: restoring the original astronomical data dimensions
        sampledImgs = inverse_astronomical_transform(sampledImgs, channels=modelConfig["selected_channel"])

        # save to .npy
        output_dir = modelConfig["sampled_dir"]
        os.makedirs(output_dir, exist_ok=True)
        np.save(
            os.path.join(output_dir, "sampled_imgs.npy"),
            sampledImgs.cpu().numpy()
        )
        print(f"Saved raw astronomical data to {output_dir}")


if __name__ == '__main__':
    modelConfig = {
        "state": "train",  # or sampling
        "epoch": 100,
        # "batch_size": 1024,
        "batch_size": 2,  # local use
        "T": 1000,
        "num_img_channel": 1,
        "selected_channel": ["z"],
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_99_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",

        # myriad use
        # "images_path": r"/home/ucaphey/Scratch/sdss.npz",
        # "conditions_path": r"/home/ucaphey/Scratch/sdss_selected_properties.csv",
        # local use
        "images_path": r"C:\Users\asus\Desktop\Files\学\UCL\Research Project\Datasets\sdss_slice.npz",
        "conditions_path": r"C:\Users\asus\Desktop\Files\学\UCL\Research Project\Datasets\sdss_selected_properties.csv",
    }

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        sampling(modelConfig)
