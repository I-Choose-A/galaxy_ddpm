import numpy as np
import torch
import torchvision.models as models
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SDSS_Single_Class, FakeData

modelConfig = {
    "batch_size": 128, # batch size
    "num_classes": 5, # number of morphological categories
    "device": "cuda:0", # computing device: gpu
    # path of data and labels
    "real_images_path": r"/home/ucaphey/Scratch/sdss.npz",
    "real_label_path": r"/home/ucaphey/Scratch/sdss_morphology_labels.csv",
    "inception_path": r"/home/ucaphey/Scratch/galaxy_ddpm/Checkpoints/best_inception_epoch.pth"
}


# load Inception-v3 model
def load_inception():
    # init inception-v3 model
    inception = models.inception_v3(weights=None, init_weights=True, aux_logits=False)
    original_conv = inception.Conv2d_1a_3x3.conv
    inception.Conv2d_1a_3x3.conv = torch.nn.Conv2d(
        in_channels=5,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    # init new weight
    torch.nn.init.kaiming_normal_(inception.Conv2d_1a_3x3.conv.weight, mode='fan_out', nonlinearity='relu')
    # replace fc with 5 class classification output
    inception.fc = torch.nn.Linear(inception.fc.in_features, modelConfig["num_classes"])
    inception.aux_logits = False
    # load trained model
    inception.load_state_dict(torch.load(modelConfig["inception_path"]))
    inception.eval()
    inception.fc = torch.nn.Identity() #remove the last classification flayer
    return inception


# get image features
def get_features(dataloader, model, device=modelConfig["device"]):
    model = model.to(device)
    model.eval()
    features = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            feat = model(images)  # [batch_size, 2048]
            features.append(feat.cpu())

    return torch.cat(features, dim=0)

# compute covariance
def torch_cov(m):
    fact = 1.0 / (m.size(0) - 1)
    m_centered = m - torch.mean(m, dim=0, keepdim=True)
    return fact * m_centered.matmul(m_centered.t()).squeeze()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.mm(sigma2).numpy())
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = torch.from_numpy(covmean).float()
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_fid(real_loader, fake_loader, device=modelConfig["device"]):
    inception = load_inception().to(device)

    # get features
    real_features = get_features(real_loader, inception, device)
    fake_features = get_features(fake_loader, inception, device)

    # calculate mean and covariance
    mu_real, sigma_real = torch.mean(real_features, dim=0), torch_cov(real_features)
    mu_fake, sigma_fake = torch.mean(fake_features, dim=0), torch_cov(fake_features)

    # calculate metrics
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid


if __name__ == '__main__':
    # dataset setup
    alpha = 0.05
    sep_mean = [0.006874967832118273, 0.03785848245024681, 0.07799547165632248, 0.1106007769703865, 0.13126179575920105]
    sep_std = [0.08303029835224152, 0.255173921585083, 0.46921947598457336, 0.6261722445487976, 0.8178646564483643]
    # use 2 set of transform for real and fake images
    astronomical_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.nan_to_num(x, nan=0.0)),
        transforms.Lambda(lambda x: np.clip(x, -0.999, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=sep_mean, std=sep_std),
        transforms.Lambda(lambda x: torch.tanh(alpha * x)),
        transforms.Resize((299, 299)),
    ])

    fake_transform = transforms.Compose([
        transforms.Resize((299, 299))
    ])

    # compute fid in class wise
    for i in range(5):
        sdss_single_class = SDSS_Single_Class(
            images_path=modelConfig["real_images_path"],
            conditions_path=modelConfig["real_label_path"],
            classification=i,
            transform=astronomical_transform
        )
        real_dataloader = DataLoader(sdss_single_class, batch_size=modelConfig["batch_size"], shuffle=False)

        fake_images = FakeData(
            images_path=rf"/home/ucaphey/Scratch/galaxy_ddpm/SampledImgs/results/generated_class_{i}_Median.npy",
            classification=i,
            transform=fake_transform
        )
        fake_dataloader = DataLoader(fake_images, batch_size=modelConfig["batch_size"], shuffle=False)

        fid_score = compute_fid(real_dataloader, fake_dataloader)
        print(f"metrics score = {fid_score}")
