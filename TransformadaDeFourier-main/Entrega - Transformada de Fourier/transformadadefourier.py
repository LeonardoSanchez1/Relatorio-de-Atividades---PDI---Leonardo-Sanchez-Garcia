import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue a imagem
imagem = cv2.imread('car.tif', cv2.IMREAD_GRAYSCALE)

# Calcule a Transformada de Fourier 2D
transformada_fourier = np.fft.fft2(imagem)
transformada_fourier_deslocada = np.fft.fftshift(transformada_fourier)  # Desloque para o centro

# Calcule o espectro de magnitude
espectro_magnitude = np.abs(transformada_fourier_deslocada)

# Calcule a fase da Transformada de Fourier
fase = np.angle(transformada_fourier_deslocada)

# Visualize a imagem original, o espectro de magnitude e a fase
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(np.log1p(espectro_magnitude), cmap='gray')
plt.title('Espectro de Magnitude')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(fase, cmap='gray')
plt.title('Fase')
plt.xticks([]), plt.yticks([])

plt.show()
