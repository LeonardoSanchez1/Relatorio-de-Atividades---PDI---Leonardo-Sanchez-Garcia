import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar filtros passa-baixa
def ideal_lowpass_filter(shape, cutoff_freq):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = np.zeros_like(D)
    H[D <= cutoff_freq] = 1
    return H

def butterworth_lowpass_filter(shape, cutoff_freq, order):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (D / cutoff_freq)**(2 * order))
    return H

def gaussian_lowpass_filter(shape, cutoff_freq):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = np.exp(-(D**2) / (2 * (cutoff_freq**2)))
    return H

# Carregando a imagem
image = cv2.imread('sinc_original_menor.tif', cv2.IMREAD_GRAYSCALE)

# Redimensionando a imagem
image = cv2.resize(image, (256, 256))

# Aplicando a Transformada de Fourier
fft_image = np.fft.fftshift(np.fft.fft2(image))
magnitude_spectrum = np.log(np.abs(fft_image) + 1)

# Definindo os parâmetros dos filtros
shape = image.shape
cutoff_freq = 50
order = 2

# Criando os filtros
ideal_filter = ideal_lowpass_filter(shape, cutoff_freq)
butterworth_filter = butterworth_lowpass_filter(shape, cutoff_freq, order)
gaussian_filter = gaussian_lowpass_filter(shape, cutoff_freq)

# Aplicando os filtros
filtered_ideal = fft_image * ideal_filter
filtered_butterworth = fft_image * butterworth_filter
filtered_gaussian = fft_image * gaussian_filter

# Calculando a transformada inversa de Fourier para as imagens filtradas
filtered_ideal = np.fft.ifft2(np.fft.ifftshift(filtered_ideal)).real
filtered_butterworth = np.fft.ifft2(np.fft.ifftshift(filtered_butterworth)).real
filtered_gaussian = np.fft.ifft2(np.fft.ifftshift(filtered_gaussian)).real

# Visualizando as imagens
plt.figure(figsize=(10, 5))

plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Imagem Original'), plt.axis('off')

plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de Fourier'), plt.axis('off')

plt.subplot(233), plt.imshow(ideal_filter, cmap='gray')
plt.title('Filtro Ideal'), plt.axis('off')

plt.subplot(234), plt.imshow(filtered_ideal, cmap='gray')
plt.title('Imagem Filtrada (Ideal)'), plt.axis('off')

plt.subplot(235), plt.imshow(butterworth_filter, cmap='gray')
plt.title('Filtro Butterworth'), plt.axis('off')

plt.subplot(236), plt.imshow(filtered_butterworth, cmap='gray')
plt.title('Imagem Filtrada (Butterworth)'), plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(231), plt.imshow(gaussian_filter, cmap='gray')
plt.title('Filtro Gaussiano'), plt.axis('off')

plt.subplot(232), plt.imshow(filtered_gaussian, cmap='gray')
plt.title('Imagem Filtrada (Gaussiano)'), plt.axis('off')

plt.tight_layout()
plt.show()


