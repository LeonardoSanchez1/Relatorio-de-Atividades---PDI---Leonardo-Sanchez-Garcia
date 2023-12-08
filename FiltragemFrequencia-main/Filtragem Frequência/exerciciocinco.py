import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('sinc_original_menor.tif', cv2.IMREAD_GRAYSCALE)

# Valores de D0 a serem testados
valores_D0 = [0.01, 0.05, 0.5]

# Loop sobre diferentes valores de D0
for D0 in valores_D0:
    # Tamanho da imagem
    rows, cols = imagem.shape

    # Criar filtros
    filtro_ideal = np.zeros((rows, cols))
    filtro_butterworth = np.zeros((rows, cols))
    filtro_gaussiano = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distancia = np.sqrt((i - rows/2)**2 + (j - cols/2)**2)
            filtro_ideal[i, j] = 1 if distancia > D0 else 0
            filtro_butterworth[i, j] = 1 / (1 + (distancia / D0)**4)  # Ordem 4
            filtro_gaussiano[i, j] = np.exp(-distancia**2 / (2 * D0**2))

    # Calcular o espectro de Fourier
    fft = np.fft.fftshift(np.fft.fft2(imagem))
    magnitude = np.log(np.abs(fft) + 1)

    # Aplicar filtros
    imagem_filtrada_ideal = np.fft.ifft2(np.fft.ifftshift(fft * filtro_ideal)).real
    imagem_filtrada_butterworth = np.fft.ifft2(np.fft.ifftshift(fft * filtro_butterworth)).real
    imagem_filtrada_gaussiano = np.fft.ifft2(np.fft.ifftshift(fft * filtro_gaussiano)).real

    # Exibir imagens filtradas
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(imagem_filtrada_ideal, cmap='gray')
    plt.title(f'Filtro Ideal (D0={D0})')

    plt.subplot(132)
    plt.imshow(imagem_filtrada_butterworth, cmap='gray')
    plt.title(f'Filtro Butterworth (D0={D0})')

    plt.subplot(133)
    plt.imshow(imagem_filtrada_gaussiano, cmap='gray')
    plt.title(f'Filtro Gaussiano (D0={D0})')

    plt.show()
