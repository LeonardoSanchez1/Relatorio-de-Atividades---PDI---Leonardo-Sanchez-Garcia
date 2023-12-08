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

# Crie uma grade de coordenadas de frequência
rows, cols = imagem.shape
freq_rows = np.fft.fftshift(np.fft.fftfreq(rows))
freq_cols = np.fft.fftshift(np.fft.fftfreq(cols))

# Crie a grade 2D de frequências com as dimensões corretas
freq_cols, freq_rows = np.meshgrid(freq_cols, freq_rows)

# Plotar o espectro 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(freq_cols, freq_rows, np.log1p(espectro_magnitude), cmap='viridis')

ax.set_xlabel('Frequência de Coluna')
ax.set_ylabel('Frequência de Linha')
ax.set_zlabel('Log(Espectro de Magnitude)')
ax.set_title('Espectro 3D da Transformada de Fourier')

plt.show()
