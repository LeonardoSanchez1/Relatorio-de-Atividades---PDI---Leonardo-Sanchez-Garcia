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
    
    # Add white squares in each corner
    npImg[0:10, 0:10] = 255  # Top-left corner
    npImg[0:10, -10:] = 255  # Top-right corner
    npImg[-10:, 0:10] = 255  # Bottom-left corner
    npImg[-10:, -10:] = 255  # Bottom-right corner
    
    # Convert numpy array back to Image
    imgNew = Image.fromarray(npImg)
    
    # Plot using matplotlib
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagem original")
    ax[1].imshow(imgNew, cmap='gray')    
    ax[1].set_title("Imagem com quadrados brancos")
    plt.show()   

if __name__ == "__main__":
    main()