import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = Image.open('house.tif')
    print(img)
    img_array= np.array(img)
    negative_image = 255 - img_array

    negative_image = Image.fromarray(negative_image.astype('uint8'))

    plt.figure(figsize=(10,5))
 

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(negative_image, cmap='gray')
    plt.title('Imagem negativa')

    plt.show()


if __name__ == "__main__":
    main()