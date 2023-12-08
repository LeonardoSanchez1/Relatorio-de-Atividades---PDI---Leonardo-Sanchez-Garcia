import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = Image.open('house.tif')
    
    # Redução em 1.5x
    reduced_img = img.resize((int(img.width / 1.5), int(img.height / 1.5)))
    
    # Aumentar em 2.5x
    enlarged_img = reduced_img.resize((int(reduced_img.width * 2.5), int(reduced_img.height * 2.5)))
    
    # Convertendo a imagem aumentada em um array NumPy
    npImg = np.array(enlarged_img)
    
    # Aplicando alguma operação no array, como definir uma região para branco
    npImg[0:100, 0:100] = 255
    
    # Convertendo o array NumPy de volta para uma imagem
    imgNew = Image.fromarray(npImg)
    
    # Plot usando Matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title("Imagem original")
    ax[0, 1].imshow(enlarged_img, cmap='gray')
    ax[0, 1].set_title("Imagem reduzida e aumentada")
    ax[1, 0].imshow(reduced_img, cmap='gray')
    ax[1, 0].set_title("Imagem reduzida")
    plt.show()

if __name__ == "__main__":
    main()
