import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('fractured_spine.tif', 0)

# Criar uma figura para exibir as imagens
plt.figure(figsize=(12, 6))

# Aplicar a transformação logarítmica com diferentes valores de c
valores_c = [10, 50, 100]  # Você pode ajustar esses valores conforme necessário

for i, c in enumerate(valores_c):
    # Aplicar a transformação logarítmica
    imagem_transformada = c * np.log1p(imagem)
    imagem_transformada = np.uint8(imagem_transformada)

    # Plotar a imagem transformada em uma grade
    plt.subplot(1, len(valores_c), i + 1)
    plt.imshow(imagem_transformada, cmap='gray')
    plt.title(f'Transformação Logarítmica (c={c})')
    plt.axis('off')

plt.tight_layout()
plt.show()
