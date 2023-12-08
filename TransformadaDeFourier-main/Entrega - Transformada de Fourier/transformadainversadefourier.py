import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue a imagem
imagem = cv2.imread('sinc.png', cv2.IMREAD_GRAYSCALE)

# Calcule a Transformada de Fourier 2D
transformada_fourier = np.fft.fft2(imagem)
transformada_fourier_deslocada = np.fft.fftshift(transformada_fourier)  # Desloque para o centro

# Calcule o espectro de magnitude
espectro_magnitude = np.abs(transformada_fourier_deslocada)

# Calcule a fase da Transformada de Fourier
fase = np.angle(transformada_fourier_deslocada)

# Calcule a Transformada Inversa de Fourier
imagem_reconstruida = np.fft.ifft2(np.fft.ifftshift(transformada_fourier_deslocada)).real

# Visualize a imagem original, o espectro de magnitude, a fase e a imagem reconstruída
plt.figure(figsize=(18, 6))

plt.subplot(141)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.xticks([]), plt.yticks([])

plt.subplot(142)
plt.imshow(np.log1p(espectro_magnitude), cmap='gray')
plt.title('Espectro de Magnitude')
plt.xticks([]), plt.yticks([])

plt.subplot(143)
plt.imshow(fase, cmap='gray')
plt.title('Fase')
plt.xticks([]), plt.yticks([])

plt.subplot(144)
plt.imshow(imagem_reconstruida, cmap='gray')
plt.title('Imagem Reconstruída')
plt.xticks([]), plt.yticks([])

plt.show()
