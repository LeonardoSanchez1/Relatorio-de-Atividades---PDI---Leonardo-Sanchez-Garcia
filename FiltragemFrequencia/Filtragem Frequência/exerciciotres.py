import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar filtros passa-alta
def ideal_highpass_filter(shape, cutoff_freq):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = np.ones_like(D)
    H[D <= cutoff_freq] = 0
    return H

def butterworth_highpass_filter(shape, cutoff_freq, order):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    
    # Evitar divisão por zero
    D[D == 0] = 1e-10  # Pequeno valor para evitar divisão por zero
    
    H = 1 / (1 + (cutoff_freq / D)**(2 * order))
    return H


def gaussian_highpass_filter(shape, cutoff_freq):
    rows, cols = shape
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 - np.exp(-(D**2) / (2 * (cutoff_freq**2)))
    return H

# Função para aplicar o filtro na frequência
def apply_filter(image, filter):
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    filtered_image = fft_image * filter
    return np.fft.ifft2(np.fft.ifftshift(filtered_image)).real

# Carregando as imagens
images = ['sinc_original_menor.tif', 'sinc_original.png', 'sinc_rot.png', 'sinc_rot2.png', 'sinc_trans.png']
image_names = ['Imagem Original', 'Espectro de Fourier', 'Filtro Ideal', 'Imagem Filtrada (Ideal)',
               'Filtro Butterworth', 'Imagem Filtrada (Butterworth)',
               'Filtro Gaussiano', 'Imagem Filtrada (Gaussiano)']

for image_path in images:
    # Carregando a imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Redimensionando a imagem
    image = cv2.resize(image, (256, 256))
    
    # Aplicando a Transformada de Fourier
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.log(np.abs(fft_image) + 1)
    
    # Definindo os parâmetros dos filtros
    shape = image.shape
    cutoff_freq = 50
    order = 2
    
    # Criando os filtros passa-alta
    ideal_filter = ideal_highpass_filter(shape, cutoff_freq)
    butterworth_filter = butterworth_highpass_filter(shape, cutoff_freq, order)
    gaussian_filter = gaussian_highpass_filter(shape, cutoff_freq)
    
    # Aplicando os filtros passa-alta
    filtered_ideal = apply_filter(image, ideal_filter)
    filtered_butterworth = apply_filter(image, butterworth_filter)
    filtered_gaussian = apply_filter(image, gaussian_filter)
    
    # Visualizando as imagens
    plt.figure(figsize=(10, 5))
    
    plt.subplot(331), plt.imshow(image, cmap='gray')
    plt.title('Imagem Original'), plt.axis('off')
    
    plt.subplot(332), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Espectro de Fourier'), plt.axis('off')
    
    plt.subplot(333), plt.imshow(ideal_filter, cmap='gray')
    plt.title('Filtro Ideal'), plt.axis('off')
    
    plt.subplot(334), plt.imshow(filtered_ideal, cmap='gray')
    plt.title('Imagem Filtrada (Ideal)'), plt.axis('off')
    
    plt.subplot(335), plt.imshow(butterworth_filter, cmap='gray')
    plt.title('Filtro Butterworth'), plt.axis('off')
    
    plt.subplot(336), plt.imshow(filtered_butterworth, cmap='gray')
    plt.title('Imagem Filtrada (Butterworth)'), plt.axis('off')
    
    plt.subplot(337), plt.imshow(gaussian_filter, cmap='gray')
    plt.title('Filtro Gaussiano'), plt.axis('off')
    
    plt.subplot(338), plt.imshow(filtered_gaussian, cmap='gray')
    plt.title('Imagem Filtrada (Gaussiano)'), plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Análise de Filtros Passa-Alta - {image_path}', fontsize=16)
    plt.show()
