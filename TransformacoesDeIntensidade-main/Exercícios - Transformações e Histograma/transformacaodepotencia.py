import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('fractured_spine.tif', 0)

# Criar uma figura para exibir as imagens
plt.figure(figsize=(12, 6))

# Aplicar a transformação de potência com diferentes valores de y
valores_y = [0.5, 1.0, 1.5]  # Você pode ajustar esses valores conforme necessário
c = 1

for i, y in enumerate(valores_y):
    # Aplicar a transformação de potência
    imagem_transformada = c * np.power(imagem, y)
    imagem_transformada = np.uint8(imagem_transformada)

    # Plotar a imagem transformada em uma grade
    plt.subplot(1, len(valores_y), i + 1)
    plt.imshow(imagem_transformada, cmap='gray')
    plt.title(f'Transformação de Potência (y={y}, c={c})')
    plt.axis('off')

plt.tight_layout()
plt.show()
