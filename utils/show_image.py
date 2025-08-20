import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_rgb, ManualInterval


def safe_arctanh(x, epsilon=1e-7):
    # clip values to prevent division by zero in arctanh
    x_clipped = np.clip(x, -1 + epsilon, 1 - epsilon)
    return np.arctanh(x_clipped)


def reverse_transforms(array, alpha=0.05):
    # apply inverse tanh
    array = safe_arctanh(array) / alpha

    # mean-std deviation values for each channel
    sep_mean = [0.006874967832118273, 0.03785848245024681, 0.07799547165632248, 0.1106007769703865, 0.13126179575920105]
    sep_std = [0.08303029835224152, 0.255173921585083, 0.46921947598457336, 0.6261722445487976, 0.8178646564483643]

    # reshape for broadcasting
    mean = np.array(sep_mean).reshape(-1, 1, 1)
    std = np.array(sep_std).reshape(-1, 1, 1)

    # reverse normalization
    denormalized = array * std + mean

    return denormalized


def show_samples():
    num_classes = 5
    num_images = 5  # show 5 real and 5 generated images per class

    fig = plt.figure(figsize=(20, 10))  # Create a wider figure

    for gclass in range(num_classes):
        # load real images
        real_imgs = np.load(
            rf"SampledImgs/real_images_samples/processed_real_images/sdss_class_{gclass}.npy"
        )
        print(f"[Class {gclass} Real] min: {np.min(real_imgs)}, max: {np.max(real_imgs)}, "
              f"avg: {np.average(real_imgs)}, median: {np.median(real_imgs)}")

        # load generated images
        fake_imgs = np.load(
            rf"SampledImgs/results/generated_class_{gclass}_Median.npy"
        )
        print(f"[Class {gclass} Fake] min: {np.min(fake_imgs)}, max: {np.max(fake_imgs)}, "
              f"avg: {np.average(fake_imgs)}, median: {np.median(fake_imgs)}")

        for i in range(num_images):
            # real image
            ax = plt.subplot(num_classes, num_images * 2, gclass * num_images * 2 + i + 1)

            # extract channels for rgb visualization
            i_channel = real_imgs[i, 3, :, :]
            r_channel = real_imgs[i, 2, :, :]
            g_channel = real_imgs[i, 1, :, :]

            # calculate maximum value for consistent scaling
            pctl = 99.3
            maximum = max(np.percentile(ch, pctl) for ch in [i_channel, r_channel, g_channel])
            rgb = make_rgb(i_channel, r_channel, g_channel,
                           interval=ManualInterval(vmin=0, vmax=maximum))

            ax.imshow(rgb)
            if i == 0:  # add class label for first image in row
                ax.set_ylabel(f"Class {gclass}", fontsize=12)
            ax.set_title(f"Real {i}", fontsize=16)
            ax.axis("off")

            # generated images
            ax = plt.subplot(num_classes, num_images * 2, gclass * num_images * 2 + num_images + i + 1)

            # extract channels for RGB visualization
            i_channel = fake_imgs[i, 3, :, :]
            r_channel = fake_imgs[i, 2, :, :]
            g_channel = fake_imgs[i, 1, :, :]

            # calculate maximum value for consistent scaling
            maximum = max(np.percentile(ch, pctl) for ch in [i_channel, r_channel, g_channel])
            rgb = make_rgb(i_channel, r_channel, g_channel,
                           interval=ManualInterval(vmin=0, vmax=maximum))

            ax.imshow(rgb)
            ax.set_title(f"Fake {i}", fontsize=16)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_samples()