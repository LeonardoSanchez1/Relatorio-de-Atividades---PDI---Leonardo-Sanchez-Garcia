import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue a imagem em escala de cinza
imagem = cv2.imread('sinc_original_menor.tif', 0)

# Calcule a Transformada de Fourier 2D da imagem
fourier = np.fft.fft2(imagem)
fourier_shifted = np.fft.fftshift(fourier)
magnitude_spectrum = np.log(np.abs(fourier_shifted) + 1)

# Valores diferentes de D0
D0_values = [0.01, 0.05, 0.5]

plt.figure(figsize=(12, 12))

for i, D0 in enumerate(D0_values):
    # Crie o filtro passa-baixa Butterworth com o valor de D0
    n = 2  # Ordem do filtro Butterworth
    filtro_butterworth = 1 / (1 + (np.sqrt(2) - 1) * np.power(magnitude_spectrum / D0, 2 * n))

    # Aplique o filtro à Transformada de Fourier
    fourier_filtrado_butterworth = fourier_shifted * filtro_butterworth

    # Calcule a Transformada Inversa de Fourier para obter a imagem filtrada
    imagem_filtrada_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_filtrado_butterworth)))

    # Visualize as imagens dos filtros e as imagens filtradas
    plt.subplot(3, 3, i * 2 + 1)
    plt.title(f'Filtro Butterworth (D0 = {D0})')
    plt.imshow(filtro_butterworth, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, i * 2 + 2)
    plt.title(f'Imagem Filtrada (D0 = {D0})')
    plt.imshow(imagem_filtrada_butterworth, cmap='gray')
    plt.axis('off')

# Visualize a imagem original
plt.subplot(3, 3, 7)
plt.title('Imagem Original')
plt.imshow(imagem, cmap='gray')
plt.axis('off')

# Visualize o espectro de Fourier
plt.subplot(3, 3, 8)
plt.title('Espectro de Fourier')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
