import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset import SDSS

modelConfig = {
    "epoch": 40, # number of training iteration
    "batch_size": 128, # batch size
    "dropout": 0.15, # prob of dropout
    "lr": 5e-6, # learning rate
    "num_classes": 5, # number of morphological categories
    "img_size": 64, # size of image
    "device": "cuda:0", # computing device: gpu
    "grad_clip": 1., # clip_grad_norm bound
    # path of data and labels
    "images_path": r"/home/ucaphey/Scratch/sdss.npz",
    "conditions_path": r"/home/ucaphey/Scratch/sdss_morphology_labels.csv",
}


def train_inception():
    device = torch.device(modelConfig["device"])

    # Dataset setup
    alpha = 0.05
    sep_mean = [0.006874967832118273, 0.03785848245024681, 0.07799547165632248, 0.1106007769703865, 0.13126179575920105]
    sep_std = [0.08303029835224152, 0.255173921585083, 0.46921947598457336, 0.6261722445487976, 0.8178646564483643]

    astronomical_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.nan_to_num(x, nan=0.0)),
        transforms.Lambda(lambda x: np.clip(x, -0.999, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=sep_mean, std=sep_std),
        transforms.Lambda(lambda x: torch.tanh(alpha * x)),
        transforms.Resize((299, 299))
    ])

    sdss_gz = SDSS(
        images_path=modelConfig["images_path"],
        conditions_path=modelConfig["conditions_path"],
        transform=astronomical_transform
    )

    train_size = int(0.7 * len(sdss_gz))
    val_size = int(0.15 * len(sdss_gz))
    test_size = len(sdss_gz) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(sdss_gz, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    inception.fc = torch.nn.Linear(inception.fc.in_features, modelConfig["num_classes"])

    # init optimizer, scheduler and loss function
    optimizer = torch.optim.AdamW(
        inception.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10,
        factor=0.8,
        min_lr=1e-6
    )
    loss_function = torch.nn.CrossEntropyLoss()

    # training process
    best_epoch = 0
    best_val_acc = 0.0
    inception = inception.to(device)
    for epoch in range(modelConfig["epoch"]):
        inception.train()
        epoch_train_loss = 0.0

        for batch_idx, (x, condition) in enumerate(train_loader):
            y = condition[:, -1].long()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = inception(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inception.parameters(), modelConfig["grad_clip"])

            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0)

            # Print batch progress every 10% of dataset
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch {epoch + 1}/{modelConfig['epoch']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Batch Loss (per sample): {loss.item():.4f} | ")

        # Validation Phase
        inception.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # validation
        with torch.no_grad():
            for x_val, condition_val in val_loader:
                y_val = condition_val[:, -1].long()
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = inception(x_val)
                loss = loss_function(outputs, y_val)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"Train Loss: {epoch_train_loss / len(train_dataset):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            torch.save(inception.state_dict(), "../checkpoints/best_inception_epoch.pth")
            print(f"Saved new best model with Val Acc: {val_acc:.2f}%")

        # Adjust learning rate
        scheduler.step(val_acc)

    print(f"best epoch is {best_epoch}")
    # Testing
    print("\nTesting on test set...")
    inception.load_state_dict(torch.load(f"../checkpoints/best_inception_epoch.pth"))  # Load best model
    test_acc = evaluate(inception, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

# testing function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, condition in dataloader:
            y = condition[:, -1].long()
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
    train_inception()
