import argparse
import os
import pickle

import numpy as np
import cv2
import torch as torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_histogram(batch_data):
    hist, _ = np.histogram(batch_data, bins=np.arange(257), density=False)
    return hist.astype(np.float32)


def gmm_noise(image_shape, gmm, device='cpu'):
    n_samples = np.prod(image_shape)
    samples, _ = gmm.sample(n_samples)
    samples = torch.tensor(samples[:, 0], dtype=torch.float32, device=device).view(*image_shape)
    return samples


def process_images(folder_path):
    data = []

    file_names = [file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]

    for file_name in tqdm(file_names, desc="Processing images"):
        image_path = os.path.join(folder_path, file_name)
        if os.path.isfile(image_path):
            img = cv2.imread(image_path, 0)
            img_data = img.ravel()
            data.extend(img_data)

    return np.array(data)

def calculate_histogram(data):
    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=254, range=(2, 256), density=True)
    return hist, bin_edges

def plot_data(hist, gmm):
    x = np.linspace(2, 255, 254)

    # Plot the original histogram
    plt.bar(x, hist, color='gray', alpha=0.5, label='Aggregated Histogram', width=1)

    if gmm is not None:
        # Calculate and plot the individual Gaussian components
        gmm_x = np.linspace(2, 255, 254)
        gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
        plt.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")

    plt.legend()
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Gaussian Mixture Model Fitting to Aggregated Histogram')
    plt.show()

    # Generate noise using the updated `gmm_noise()` function
    image_shape = (64, 1, 64, 64)  # Replace with the desired shape of your noise tensor
    noise = gmm_noise(image_shape, gmm)

    # Rescale the noise to the range [0, 255] and convert it to an 8-bit unsigned integer
    first_noise = noise[0, :, :, :].squeeze().cpu().numpy()
    first_noise_rescaled = ((first_noise - first_noise.min()) / (first_noise.max() - first_noise.min()) * 255).astype(
        np.uint8)

    # Display the noise image using `plt.imshow()`
    plt.imshow(first_noise_rescaled)
    plt.axis('off')
    plt.title('First Noise Sample as an Image')
    plt.show()

def save_to_file(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit GMM to images and plot histogram.')
    parser.add_argument('folder_path', help='Path to the folder containing images')
    parser.add_argument('--save_histogram', type=str, nargs='?', const='histogram.pkl',
                        help='Path to save the aggregated histogram to a file.')
    parser.add_argument('--save_gaussian_mixture', type=str, nargs='?', const='gmm.pkl',
                        help='Path to save the Gaussian Mixture Model to a file.')

    args = parser.parse_args()

    data = process_images(args.folder_path)

    hist, bin_edges = calculate_histogram(data)

    # Fit GMM
    gmm = GaussianMixture(n_components=6)
    gmm.fit(data.reshape(-1, 1))

    plot_data(hist, gmm)

    if args.save_histogram:
        save_to_file(data, args.save_histogram)

    if args.save_gaussian_mixture:
        save_to_file(gmm, args.save_gaussian_mixture)



