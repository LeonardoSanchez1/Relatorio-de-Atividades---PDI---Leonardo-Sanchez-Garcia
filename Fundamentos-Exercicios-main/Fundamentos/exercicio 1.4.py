import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = Image.open('house.tif')
    print(img.format)
    print(img.size)
    print(img.mode)
    
    # Convert image to numpy array
    npImg = np.array(img)
    
    # Add black square in the center
    center_x = npImg.shape[0] // 2
    center_y = npImg.shape[1] // 2
    half_size = 15 // 2
    npImg[center_x - half_size: center_x + half_size, center_y - half_size: center_y + half_size] = 0
    
    # Convert numpy array back to Image
    imgNew = Image.fromarray(npImg)
    
    # Plot using matplotlib
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow(imgNew, cmap='gray')    
    ax[1].set_title("Imagem com quadrado preto no centro")
    plt.show()   

if __name__ == "__main__":
    main()