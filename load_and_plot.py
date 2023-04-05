import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def load_data(hist_file=None, gmm_file=None):
    histogram, gmm = None, None

    if hist_file:
        with open(hist_file, 'rb') as f:
            histogram = pickle.load(f)

            # Normalize the aggregated histogram
            histogram = histogram / histogram.sum()

    if gmm_file:
        with open(gmm_file, 'rb') as f:
            gmm = pickle.load(f)

    return histogram, gmm

def plot_data(histogram=None, gmm=None):
    x = np.arange(0, 256)

    if histogram is not None:
        # Plot the original histogram
        plt.bar(x, histogram, color='gray', alpha=0.5, label='Aggregated Histogram')

    if gmm is not None:
        # Calculate and plot the individual Gaussian components
        n_components = gmm.n_components
        means = gmm.means_[:, 0]
        weights = gmm.weights_
        covars = gmm.covariances_[:, 0]

        for i in range(n_components):
            gauss = weights[i] * (1 / np.sqrt(2 * np.pi * covars[i])) * np.exp(-(x - means[i]) ** 2 / (2 * covars[i]))
            plt.plot(x, gauss * histogram.sum(), label=f'Gaussian {i + 1}')

    plt.legend()
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Gaussian Mixture Model Fitting to Aggregated Histogram')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and plot histogram and/or Gaussian Mixture Model components.')
    parser.add_argument('--histogram_file', type=str, help='Path to the aggregated histogram file.')
    parser.add_argument('--gmm_file', type=str, help='Path to the Gaussian Mixture Model file.')
    parser.add_argument('--calculate_gaussian', action='store_true', help='Calculate Gaussian components from the histogram.')

    args = parser.parse_args()

    histogram, gmm = load_data(args.histogram_file, args.gmm_file)

    if args.calculate_gaussian and histogram is not None:
        n_components = 4
        X = np.column_stack((np.arange(0, 256), histogram))
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.fit(X)

    plot_data(histogram, gmm)
