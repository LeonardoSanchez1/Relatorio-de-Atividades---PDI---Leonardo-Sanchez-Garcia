import numpy as np
import cv2
from matplotlib import pyplot as plt

# Função para aplicar a convolução com uma máscara em uma imagem
def apply_convolution(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Crie uma imagem de saída com o mesmo tamanho da imagem original
    output = np.zeros_like(image)

    # Faça a convolução manualmente
    for y in range(pad_height, height - pad_height):
        for x in range(pad_width, width - pad_width):
            region = image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
            output[y, x] = np.sum(region * kernel)

    return output

# Carregar as imagens
biel = cv2.imread('biel.png', cv2.IMREAD_GRAYSCALE)
lena = cv2.imread('lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)
cameraman = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Definir as máscaras
mascara_media = np.ones((3, 3), dtype=np.float32) / 9.0
mascara_gaussiana = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
mascara_laplaciana = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
mascara_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
mascara_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Aplicar convolução com as máscaras nas imagens
biel_media = apply_convolution(biel, mascara_media)
biel_gaussiana = apply_convolution(biel, mascara_gaussiana)
biel_laplaciana = apply_convolution(biel, mascara_laplaciana)
biel_sobel_x = apply_convolution(biel, mascara_sobel_x)
biel_sobel_y = apply_convolution(biel, mascara_sobel_y)
biel_gradiente = biel_sobel_x + biel_sobel_y
biel_laplaciano_somado = biel + biel_laplaciana

# Repetir o processo para a imagem "lena"
lena_media = apply_convolution(lena, mascara_media)
lena_gaussiana = apply_convolution(lena, mascara_gaussiana)
lena_laplaciana = apply_convolution(lena, mascara_laplaciana)
lena_sobel_x = apply_convolution(lena, mascara_sobel_x)
lena_sobel_y = apply_convolution(lena, mascara_sobel_y)
lena_gradiente = lena_sobel_x + lena_sobel_y
lena_laplaciano_somado = lena + lena_laplaciana

# Repetir o processo para a imagem "cameraman"
cameraman_media = apply_convolution(cameraman, mascara_media)
cameraman_gaussiana = apply_convolution(cameraman, mascara_gaussiana)
cameraman_laplaciana = apply_convolution(cameraman, mascara_laplaciana)
cameraman_sobel_x = apply_convolution(cameraman, mascara_sobel_x)
cameraman_sobel_y = apply_convolution(cameraman, mascara_sobel_y)
cameraman_gradiente = cameraman_sobel_x + cameraman_sobel_y
cameraman_laplaciano_somado = cameraman + cameraman_laplaciana

# Mostrar as imagens resultantes
plt.figure(figsize=(15, 10))

# Imagens "biel"
plt.subplot(3, 4, 1), plt.imshow(biel, cmap='gray'), plt.title('Original (Biel)')
plt.subplot(3, 4, 2), plt.imshow(biel_media, cmap='gray'), plt.title('Média')
plt.subplot(3, 4, 3), plt.imshow(biel_gaussiana, cmap='gray'), plt.title('Gaussiana')
plt.subplot(3, 4, 4), plt.imshow(biel_laplaciana, cmap='gray'), plt.title('Laplaciana')
plt.subplot(3, 4, 5), plt.imshow(biel_sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(3, 4, 6), plt.imshow(biel_sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(3, 4, 7), plt.imshow(biel_gradiente, cmap='gray'), plt.title('Gradiente')
plt.subplot(3, 4, 8), plt.imshow(biel_laplaciano_somado, cmap='gray'), plt.title('Laplaciano + Original')

# Imagens "lena"
plt.figure(figsize=(15, 10))
plt.subplot(3, 4, 1), plt.imshow(lena, cmap='gray'), plt.title('Original (Lena)')
plt.subplot(3, 4, 2), plt.imshow(lena_media, cmap='gray'), plt.title('Média')
plt.subplot(3, 4, 3), plt.imshow(lena_gaussiana, cmap='gray'), plt.title('Gaussiana')
plt.subplot(3, 4, 4), plt.imshow(lena_laplaciana, cmap='gray'), plt.title('Laplaciana')
plt.subplot(3, 4, 5), plt.imshow(lena_sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(3, 4, 6), plt.imshow(lena_sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(3, 4, 7), plt.imshow(lena_gradiente, cmap='gray'), plt.title('Gradiente')
plt.subplot(3, 4, 8), plt.imshow(lena_laplaciano_somado, cmap='gray'), plt.title('Laplaciano + Original')

# Imagens "cameraman"
plt.figure(figsize=(15, 10))
plt.subplot(3, 4, 1), plt.imshow(cameraman, cmap='gray'), plt.title('Original (Cameraman)')
plt.subplot(3, 4, 2), plt.imshow(cameraman_media, cmap='gray'), plt.title('Média')
plt.subplot(3, 4, 3), plt.imshow(cameraman_gaussiana, cmap='gray'), plt.title('Gaussiana')
plt.subplot(3, 4, 4), plt.imshow(cameraman_laplaciana, cmap='gray'), plt.title('Laplaciana')
plt.subplot(3, 4, 5), plt.imshow(cameraman_sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(3, 4, 6), plt.imshow(cameraman_sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(3, 4, 7), plt.imshow(cameraman_gradiente, cmap='gray'), plt.title('Gradiente')
plt.subplot(3, 4, 8), plt.imshow(cameraman_laplaciano_somado, cmap='gray'), plt.title('Laplaciano + Original')

# Ajustar o layout das janelas
plt.tight_layout()
plt.show()

