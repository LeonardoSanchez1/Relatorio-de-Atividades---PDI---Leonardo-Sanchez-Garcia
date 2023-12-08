import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('fractured_spine.tif', 0)  # 0 para carregar a imagem em escala de cinza

# Obter as dimensões da imagem
altura, largura = imagem.shape

# Criar uma lista para armazenar os planos de bits
planos_de_bits = []

# Extrair cada plano de bits
for i in range(8):  # Vamos considerar 8 bits, mas você pode ajustar conforme necessário
    plano = (imagem >> i) & 1  # Extrair o bit de cada pixel
    planos_de_bits.append(plano)

# Configurar a grade para exibir as imagens
linhas = 2
colunas = 4

# Criar uma figura para exibir as imagens
fig, axs = plt.subplots(linhas, colunas, figsize=(12, 6))

# Exibir os planos de bits na grade
for i, plano in enumerate(planos_de_bits):
    row = i // colunas
    col = i % colunas
    axs[row, col].imshow(plano, cmap='gray')
    axs[row, col].set_title(f'Plano {i}')
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
