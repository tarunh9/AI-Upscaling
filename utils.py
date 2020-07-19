import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(90, 45))

    images = [lr, sr]
    titles = [f'Original Image', f'AI Upscaled Image']
    # titles = ['.', '.']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=90)
        plt.xticks([])
        plt.yticks([])
