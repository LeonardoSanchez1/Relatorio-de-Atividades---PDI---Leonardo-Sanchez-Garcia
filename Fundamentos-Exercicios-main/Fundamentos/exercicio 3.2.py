import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def rotate_numpy(image, angle):
    rotated_image = ndimage.rotate(image, angle, reshape=False)
    return rotated_image

def rotate_pillow(image, angle):
    pil_image = Image.fromarray(image)
    rotated_pil_image = pil_image.rotate(angle, resample=Image.BILINEAR, expand=True)
    rotated_image = np.array(rotated_pil_image)
    return rotated_image

def rotate_opencv(image, angle):
    height, width = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_scipy(image, angle):
    rotated_image = ndimage.rotate(image, angle, reshape=False)
    return rotated_image

def main():
    img = Image.open('house.tif')
    npImg = np.array(img)
    npImg[0:100, 0:100] = 255

    rotation_angles = [45, 90, 100]

    fig, axes = plt.subplots(nrows=len(rotation_angles), ncols=5, figsize=(15, 5 * len(rotation_angles)))

    for i, angle in enumerate(rotation_angles):
        img_rotated_45 = rotate_pillow(npImg, angle)
        img_rotated_90 = rotate_pillow(npImg, angle)
        img_rotated_100 = rotate_pillow(npImg, angle)
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title("Original")
        
        axes[i, 1].imshow(img_rotated_45, cmap='gray')
        axes[i, 1].set_title(f"Rot. {angle}° - Pillow")
        
        axes[i, 2].imshow(rotate_opencv(npImg, angle), cmap='gray')
        axes[i, 2].set_title(f"Rot. {angle}° - OpenCV")
        
        axes[i, 3].imshow(rotate_numpy(npImg, angle), cmap='gray')
        axes[i, 3].set_title(f"Rot. {angle}° - Numpy")
        
        axes[i, 4].imshow(rotate_scipy(npImg, angle), cmap='gray')
        axes[i, 4].set_title(f"Rot. {angle}° - Scipy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
