import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def translate_numpy(image, dx, dy):
    translated_image = ndimage.shift(image, (dy, dx), mode='constant', cval=0)
    return translated_image

def translate_pillow(image, dx, dy):
    pil_image = Image.fromarray(image)
    translated_pil_image = pil_image.transform(image.shape, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    translated_image = np.array(translated_pil_image)
    return translated_image

def translate_opencv(image, dx, dy):
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

def translate_scipy(image, dx, dy):
    translated_image = ndimage.shift(image, (dy, dx), mode='constant', cval=0)
    return translated_image

def main():
    img = Image.open('house.tif')
    npImg = np.array(img)
    npImg[0:100, 0:100] = 255

    dx, dy = 35, 45  # Parâmetros de deslocamento

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))  # Adjusted ncols to 5

    fig.suptitle("Translação da Imagem", fontsize=16)

    axes[0].imshow(npImg, cmap='gray')
    axes[0].set_title("Original")

    axes[1].imshow(translate_pillow(npImg, dx, dy), cmap='gray')
    axes[1].set_title(f"Transl. ({dx}, {dy}) - Pillow")

    axes[2].imshow(translate_opencv(npImg, dx, dy), cmap='gray')
    axes[2].set_title(f"Transl. ({dx}, {dy}) - OpenCV")

    axes[3].imshow(translate_numpy(npImg, dx, dy), cmap='gray')
    axes[3].set_title(f"Transl. ({dx}, {dy}) - Numpy")

    axes[4].imshow(translate_scipy(npImg, dx, dy), cmap='gray')
    axes[4].set_title(f"Transl. ({dx}, {dy}) - Scipy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()