# Exemplo de leitura e plot de imagens
# Bibliotecas Numpy, Pillow, Matplotlib

import numpy as np
from PIL import Image # pillow
import matplotlib.pyplot as plt

def main():
    img = Image.open('imgs/lena_gray_512.tif')
    print(img.format)
    print(img.size)
    print(img.mode)
    #img.show() # Plot image using Pillow
    # converte image to numpy array
    npImg = np.array(img)
    npImg[0:100,0:100] = 255
    # converte numpy array to Image
    imgNew = Image.fromarray(npImg)
    #imgNew.show()
    # Plot using matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title("Imagem original")
    ax[0,1].imshow(imgNew, cmap='gray')    
    ax[0,1].set_title("Imagem filtrada")
    plt.show()   

if __name__ == "__main__":
    main()