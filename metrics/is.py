import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FakeData_for_IS

modelConfig = {
    "batch_size": 128, # batch size
    "num_classes": 5, # number of morphological categories
    "device": "cuda:0", # computing device: gpu
    # path of data and labels
    "images_path": r"C:/home/ucaphey/Scratch/generated_5000damples.npy",
    "inception_path": r"/home/ucaphey/Scratch/galaxy_ddpm/Checkpoints/best_inception_epoch.pth"
}

# load inception for is computing IS score
def load_inception(num_classes=5, device="cuda"):
    inception = models.inception_v3(weights=None, aux_logits=False)
    # modify the input channels
    original_conv = inception.Conv2d_1a_3x3.conv
    inception.Conv2d_1a_3x3.conv = torch.nn.Conv2d(
        in_channels=5,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    torch.nn.init.kaiming_normal_(inception.Conv2d_1a_3x3.conv.weight, mode='fan_out', nonlinearity='relu')

    # modify fc layer for 5 categories
    inception.fc = torch.nn.Linear(inception.fc.in_features, num_classes)
    inception.load_state_dict(torch.load(modelConfig["inception_path"]))
    inception.to(device)
    inception.eval()
    return inception

# compute the category probability distribution for all samples.
def compute_probs(dataloader, model, device="cuda"):
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            logits = model(images)
            probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs, dim=0)

# compute IS score
def calculate_inception_score(probs, splits=10):
    probs = np.clip(probs, 1e-16, 1.0)
    scores = []
    N = probs.shape[0]
    for i in range(splits):
        subset = probs[i * (N // splits): (i + 1) * (N // splits)]
        p_y = subset.mean(axis=0)  # marginal distribution p(y)
        kl_divs = [entropy(subset[j], p_y, base=2) for j in range(subset.shape[0])]
        avg_kl = np.mean(kl_divs)
        scores.append(np.exp(avg_kl))
    return np.mean(scores), np.std(scores)  # mean and std of IS

# main function for compute IS
def compute_is(fake_loader, device="cuda"):
    inception = load_inception(num_classes=modelConfig["num_classes"], device=device)
    probs = compute_probs(fake_loader, inception, device)
    is_mean, is_std = calculate_inception_score(probs.numpy())
    return is_mean, is_std


if __name__ == '__main__':
    # transform for generated data
    fake_transform = transforms.Compose([
        transforms.Resize((299, 299))
    ])
    # 5000
    fake_images = FakeData_for_IS(
        images_path=modelConfig["images_path"],
        transform=fake_transform
    )
    fake_dataloader = DataLoader(fake_images, batch_size=modelConfig["batch_size"], shuffle=False)

    is_mean, is_std = compute_is(fake_dataloader)
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
