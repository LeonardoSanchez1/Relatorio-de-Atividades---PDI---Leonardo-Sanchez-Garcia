import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = Image.open('house.tif')
    print(img)
    img_array= np.array(img)
    
    # Dividindo os valores dos pixels pela metade
    half_intensity_image = img_array // 2
    
    half_intensity_image = Image.fromarray(half_intensity_image.astype('uint8'))

    plt.figure(figsize=(10,5))
 
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(half_intensity_image, cmap='gray')
    plt.title('Imagem com intensidade reduzida')

    plt.show()

if __name__ == "__main__":
    main()