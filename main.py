import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from unet import UNet
from scheduler import GradualWarmupScheduler
from dataset import SDSS

modelConfig = {
    "state": "sampling",  # or sampling
    "epoch": 100,
    "batch_size": 1024,
    "T": 1000,
    "num_img_channel": 5,
    "selected_channel": ["u", "g", "r", "i", "z"],
    "channel": 64,
    "num_classes": 5,
    "channel_mult": [1, 2, 3, 4],
    "num_res_blocks": 1,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 64,
    "grad_clip": 1.,
    "device": "cuda:0",
    "training_load_weight": "ckpt_29_5ch_directly_64baseCh_arcsinh.pt",
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_29_5ch_directly_64baseCh_arcsinh_conditional.pt",
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",

    # myriad use
    "images_path": r"/home/ucaphey/Scratch/sdss.npz",
    "conditions_path": r"/home/ucaphey/Scratch/sdss_morphology_labels.csv",
}


def train():
    device = torch.device(modelConfig["device"])

    # Dataset setup
    alpha = 0.05
    astronomical_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.nan_to_num(x, nan=0.0)),
        transforms.Lambda(lambda x: np.clip(x, -0.999, 1000)),
        transforms.Lambda(lambda x: np.arcsinh(alpha * x)),
        transforms.ToTensor(),
        # arcsinh normalization
        transforms.Normalize(
            mean=[0.0003766524896491319, 0.0012405638117343187, 0.002521686488762498, 0.003659023903310299,
                  0.004779669921845198],
            std=[0.004004077520221472, 0.009601526893675327, 0.01590861566364765, 0.020500242710113525,
                 0.026334315538406372])
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
        dropout=modelConfig["dropout"],
        num_classes=modelConfig["num_classes"]
    ).to(device)

    if modelConfig["training_load_weight"] is not None:
        pretrained_dict = torch.load(
            os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"]),
            map_location=device
        )
        model_dict = net_model.state_dict()

        #  filter the layers according to condition embedding
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and "condition_embedding" not in k
        }
        model_dict.update(pretrained_dict)
        net_model.load_state_dict(model_dict, strict=False)
        print(f"Loaded pretrained weights from {modelConfig['training_load_weight']}")

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

        for batch_idx, (images, conditions) in enumerate(dataloader):
            optimizer.zero_grad()
            x_0 = images.to(device)
            c = conditions.to(device)
            loss = trainer(x_0, c).mean()
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
                      f"Batch Loss (per sample): {loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")

        warmUpScheduler.step()

        # Epoch summary
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1} completed | "
              f"Avg Loss (per sample): {avg_epoch_loss:.4f} | "
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
    # arcsinh normalization
    sdss_mean = [0.0003766524896491319, 0.0012405638117343187, 0.002521686488762498, 0.003659023903310299,
                 0.004779669921845198]
    sdss_std = [0.004004077520221472, 0.009601526893675327, 0.01590861566364765, 0.020500242710113525,
                0.026334315538406372]

    # inverse normalization
    mean = torch.tensor([sdss_mean[i] for i in channels], device=tensor.device)
    std = torch.tensor([sdss_std[i] for i in channels], device=tensor.device)
    denormalized = tensor * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    # inverse arcsinh
    alpha = 0.05
    denormalized = torch.sinh(denormalized) / alpha

    return denormalized


def sampling():
    # load model and sampling
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(
            T=modelConfig["T"],
            img_ch=modelConfig["num_img_channel"],
            ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=0.,
            num_classes=modelConfig["num_classes"]
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

        real_images_per_class = [36478, 37912, 4005, 15480, 82730]
        batch_size = modelConfig["batch_size"]
        output_dir = os.path.join(modelConfig["sampled_dir"], "fid_batches")
        os.makedirs(output_dir, exist_ok=True)

        for class_id, num_real_images in enumerate(real_images_per_class):
            print(f"\nGenerating {num_real_images} images for class {class_id}...")
            num_batches = (num_real_images + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                # adjust for final batch
                current_batch_size = min(
                    batch_size,
                    num_real_images - batch_idx * batch_size
                )

                # generate noise and conditions
                noise = torch.randn(
                    [current_batch_size, modelConfig["num_img_channel"], 64, 64],
                    device=device
                )
                c = torch.tensor([class_id] * current_batch_size, device=device)

                # sampling
                sampled = sampler(noise, c)
                sampled = inverse_astronomical_transform(
                    sampled,
                    modelConfig["selected_channel"]
                )

                # save batch to temp file
                batch_path = os.path.join(
                    output_dir,
                    f"class_{class_id}_batch_{batch_idx:04d}.npy"  # Zero-padded naming
                )
                np.save(batch_path, sampled.cpu().numpy())

                # clear cache every 5 batches
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\nOrganizing files for cFID evaluation...")
        final_output_dir = os.path.join(modelConfig["sampled_dir"], "cFID_results")
        os.makedirs(final_output_dir, exist_ok=True)

        for class_id in range(modelConfig["num_classes"]):
            # load all batches for current class
            class_files = sorted([
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith(f"class_{class_id}_") and f.endswith(".npy")
            ])

            # concatenate and save as single .npy file
            class_images = np.concatenate(
                [np.load(f) for f in class_files],
                axis=0
            )
            np.save(
                os.path.join(final_output_dir, f"generated_class_{class_id}.npy"),
                class_images
            )
            print(f"Class {class_id}: Save {len(class_images)} images")


if __name__ == '__main__':
    if modelConfig["state"] == "train":
        train()
    else:
        sampling()
