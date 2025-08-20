import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from models.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from models.unet import UNet
from utils.dataset import SDSS

modelConfig = {
    "state": "train",  # or sampling
    "epoch": 100, # number of training iteration
    "batch_size": 128, # batch size
    "T": 1000, # total time step T
    "num_img_channel": 5, # number of image channels
    "selected_channel": ["u", "g", "r", "i", "z"], # training model for which channels
    "channel": 256, # number of channels for resnet block's kernels
    "num_classes": 5, # number of morphological categories
    "num_features": 5, # number of physical features
    "channel_mult": [1, 2, 3, 4], # factor of channels number for U-Net
    "num_res_blocks": 1, # number of resnet block in a down/upsampling module
    "dropout": 0.1, # prob of dropout rate
    "lr": 5e-6, # learning rate
    "beta_1": 1e-4, # hyperparameter of beta_1
    "beta_T": 0.02, # hyperparameter of beta_T
    "img_size": 64, # size of image
    "grad_clip": 1., # clip_grad_norm bound
    "device": "cuda:0", # computing device: gpu
    # path of loading pretrained model, saved place and dataset path
    "training_load_weight": None,
    "save_weight_dir": r"checkpoints/",
    "test_load_weight": None,
    "sampled_dir": r"sampled_images/",
    "images_path": r"data_files/sdss.npz",
    "conditions_path": r"data_files/sdss_morphology_labels.csv",
}


def train():
    device = torch.device(modelConfig["device"])

    # Dataset setup
    alpha = 0.05
    sep_mean = [0.006874967832118273, 0.03785848245024681, 0.07799547165632248, 0.1106007769703865, 0.13126179575920105]
    sep_std = [0.08303029835224152, 0.255173921585083, 0.46921947598457336, 0.6261722445487976, 0.8178646564483643]

    # remove bad data -> norm -> tanh
    astronomical_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.nan_to_num(x, nan=0.0)),
        transforms.Lambda(lambda x: np.clip(x, -0.999, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=sep_mean, std=sep_std),
        transforms.Lambda(lambda x: torch.tanh(alpha * x)),
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
        num_classes=modelConfig["num_classes"],
        num_features=modelConfig["num_features"]
    ).to(device)

    # load pretrained model
    if modelConfig["training_load_weight"] is not None:
        pretrained_dict = torch.load(
            os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"]),
            map_location=device
        )
        net_model.load_state_dict(pretrained_dict)

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )

    # init trainer class for training ddpm
    trainer = GaussianDiffusionTrainer(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"]
    ).to(device)

    # Training process
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

        # warmUpScheduler.step()

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
            num_classes=modelConfig["num_classes"],
            num_features=modelConfig["num_features"]
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

        # 1/8 of real dataset
        real_images_per_class = [9120, 4978, 1000, 3870, 20682]

        batch_size = modelConfig["batch_size"]
        output_dir = os.path.join(modelConfig["sampled_dir"], "batches")
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

                # median value of each physical feature
                c = torch.tensor([3.8961502088838444, 0.008839854973913385, -2.7289007776677927, 0.07603080570697784,
                                  10.844648974261666, class_id] * current_batch_size, device=device)


                c = c.view(current_batch_size, -1)
                # sampling
                sampled = sampler(noise, c)

                # save batch to temp file
                batch_path = os.path.join(
                    output_dir,
                    f"class_{class_id}_batch_{batch_idx:04d}_Median.npy"  # Zero-padded naming
                )
                np.save(batch_path, sampled.cpu().numpy())

                # clear cache every 5 batches
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\nOrganizing files for cFID evaluation...")
        final_output_dir = os.path.join(modelConfig["sampled_dir"], "results")
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
                os.path.join(final_output_dir, f"generated_class_{class_id}_Median.npy"),
                class_images
            )
            print(f"Class {class_id}: Save {len(class_images)} images")


if __name__ == '__main__':
    if modelConfig["state"] == "train":
        train()
    else:
        sampling()
