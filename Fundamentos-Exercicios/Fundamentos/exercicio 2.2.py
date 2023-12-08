import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def apply_median_filter(image):
    filtered_image = ndimage.median_filter(image, size=3)  # 3x3 median filter
    return filtered_image.astype(np.uint8)

def main():
    img = Image.open('house.tif')
    print(img.format)
    print(img.size)
    print(img.mode)
    
    # Convert image to numpy array
    npImg = np.array(img)
    
    # Apply median filter using Pillow
    filtered_img_pillow = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Convert image to numpy array
    npImg_filtered_pillow = np.array(filtered_img_pillow)
    
    # Apply median filter using OpenCV
    filtered_img_opencv = cv2.medianBlur(npImg, 3)
    
    # Apply median filter using NumPy
    filtered_img_numpy = apply_median_filter(npImg)
    
    # Apply median filter using SciPy
    filtered_img_scipy = apply_median_filter(npImg)
    
    # Convert numpy arrays back to Images
    imgNew_pillow = Image.fromarray(npImg_filtered_pillow)
    imgNew_opencv = Image.fromarray(filtered_img_opencv)
    imgNew_numpy = Image.fromarray(filtered_img_numpy)
    imgNew_scipy = Image.fromarray(filtered_img_scipy)
    
    # Plot using matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    ax[0, 0].imshow(npImg, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[0, 1].imshow(imgNew_pillow, cmap='gray')    
    ax[0, 1].set_title("Filtered Image (Pillow)")
    ax[0, 2].imshow(imgNew_opencv, cmap='gray')    
    ax[0, 2].set_title("Filtered Image (OpenCV)")
    ax[1, 0].imshow(imgNew_numpy, cmap='gray')    
    ax[1, 0].set_title("Filtered Image (NumPy)")
    ax[1, 1].imshow(imgNew_scipy, cmap='gray')    
    ax[1, 1].set_title("Filtered Image (SciPy)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
