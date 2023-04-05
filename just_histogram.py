import os
import cv2
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_images(folder_path, search_subfolders=False):
    aggregated_hist = np.zeros(256, dtype=np.float32)

    for root, _, file_names in os.walk(folder_path):
        for file_name in tqdm(file_names, desc="Processing images"):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file_name)
                img = cv2.imread(image_path, 0)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                aggregated_hist += hist.ravel()

        if not search_subfolders:
            break

    return aggregated_hist

def save_histogram(histogram, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(histogram, file)

def draw_histogram(histogram):
    plt.bar(np.arange(256), histogram, color='gray', alpha=0.5)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Aggregated Histogram')
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate and save the aggregated histogram of image data.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("--save_histogram", type=str, default="histogram.pkl", help="File name to save the computed histogram.")
    parser.add_argument("--subfolders", action="store_true", help="Search for images in subfolders.")
    parser.add_argument("--show_histogram", action="store_true", help="Display the aggregated histogram.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    folder_path = args.folder_path
    save_histogram_file = args.save_histogram
    search_subfolders = args.subfolders
    show_histogram = args.show_histogram

    aggregated_hist = process_images(folder_path, search_subfolders)
    save_histogram(aggregated_hist, save_histogram_file)

    draw_histogram(aggregated_hist)